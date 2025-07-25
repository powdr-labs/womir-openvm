use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionBus, ExecutionState,
        MinimalInstruction, Result, VmAdapterAir, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryController, OfflineMemory, RecordId,
        },
        program::ProgramBus,
    },
};
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::ColumnsAir,
};
use openvm_womir_transpiler::JaafOpcode;
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{AdapterRuntimeContextWom, FrameBridge, FrameBus, FrameState, VmAdapterChipWom};

use super::{abstract_compose, RV32_REGISTER_NUM_LIMBS};

// This adapter reads from [b:4]_d (rs1) and writes to [a:4]_d (rd)
#[derive(Debug)]
pub struct JaafAdapterChipWom<F: Field> {
    pub air: JaafAdapterAirWom,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> JaafAdapterChipWom<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
    ) -> Self {
        Self {
            air: JaafAdapterAirWom {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                frame_bridge: FrameBridge::new(frame_bus),
                memory_bridge,
            },
            _marker: PhantomData,
        }
    }
}
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaafReadRecord {
    pub rs1: Option<RecordId>,
    pub rs2: RecordId, // Always present since FP is always needed
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaafWriteRecord {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    pub rd1_id: Option<RecordId>,
    pub rd2_id: Option<RecordId>,
    pub imm: u32,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct JaafAdapterColsWom<T> {
    pub from_state: ExecutionState<T>,
    pub from_frame: FrameState<T>,
    pub rs1_ptr: T,
    pub rs1_aux_cols: MemoryReadAuxCols<T>,
    pub rs2_ptr: T,
    pub rs2_aux_cols: MemoryReadAuxCols<T>,
    pub rd1_ptr: T,
    pub rd1_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    pub rd2_ptr: T,
    pub rd2_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    /// Only writes rd1 if `needs_write_rd1`
    pub needs_write_rd1: T,
    /// Only writes rd2 if `needs_write_rd2`
    pub needs_write_rd2: T,
    /// 1 if we need to read rs1 (for ret and call_indirect)
    pub needs_read_rs1: T,
    /// 1 if we need to read rs2 (for all opcodes)
    pub needs_read_rs2: T,
    /// Immediate value from instruction
    pub imm: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct JaafAdapterAirWom {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) frame_bridge: FrameBridge,
}

impl<F: Field> BaseAir<F> for JaafAdapterAirWom {
    fn width(&self) -> usize {
        JaafAdapterColsWom::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for JaafAdapterAirWom {
    fn columns(&self) -> Option<Vec<String>> {
        JaafAdapterColsWom::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for JaafAdapterAirWom {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        2,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &JaafAdapterColsWom<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let write_count_rd1 = local_cols.needs_write_rd1;
        let write_count_rd2 = local_cols.needs_write_rd2;
        let read_count_rs1 = local_cols.needs_read_rs1;
        let read_count_rs2 = local_cols.needs_read_rs2;

        builder.assert_bool(write_count_rd1);
        builder.assert_bool(write_count_rd2);
        builder.assert_bool(read_count_rs1);
        builder.assert_bool(read_count_rs2);
        builder
            .when::<AB::Expr>(not(ctx.instruction.is_valid.clone()))
            .assert_zero(write_count_rd1);
        builder
            .when::<AB::Expr>(not(ctx.instruction.is_valid.clone()))
            .assert_zero(write_count_rd2);

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rs1_ptr,
                ),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local_cols.rs1_aux_cols,
            )
            .eval(builder, read_count_rs1);

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rs2_ptr,
                ),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local_cols.rs2_aux_cols,
            )
            .eval(builder, read_count_rs2);

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rd1_ptr,
                ),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &local_cols.rd1_aux_cols,
            )
            .eval(builder, write_count_rd1);

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rd2_ptr,
                ),
                ctx.writes[1].clone(),
                timestamp_pp(),
                &local_cols.rd2_aux_cols,
            )
            .eval(builder, write_count_rd2);

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP));
        // The adapter will handle to_fp from reads[1] (rs2)
        // rs2 contains the target fp value
        let to_fp = abstract_compose(ctx.reads[1].clone());

        // Update the frame bridge
        self.frame_bridge.frame_bus.execute(
            builder,
            ctx.instruction.is_valid.clone(),
            local_cols.from_frame,
            FrameState {
                pc: to_pc.clone(),
                fp: to_fp,
            },
        );

        // regardless of `needs_write`, must always execute instruction when `is_valid`.
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rd1_ptr.into(),
                    local_cols.rd2_ptr.into(),
                    local_cols.rs1_ptr.into(),
                    local_cols.imm.into(),
                    local_cols.rs2_ptr.into(),
                    write_count_rd1.into(),
                    AB::Expr::ZERO,
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp + AB::F::from_canonical_usize(timestamp_delta),
                },
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &JaafAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for JaafAdapterChipWom<F> {
    type ReadRecord = JaafReadRecord;
    type WriteRecord = JaafWriteRecord;
    type Air = JaafAdapterAirWom;
    type Interface = BasicAdapterInterface<
        F,
        MinimalInstruction<F>,
        2,
        2,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;
    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        fp: u32,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { c, e, opcode, .. } = *instruction;

        let local_opcode =
            JaafOpcode::from_usize(opcode.local_opcode_idx(JaafOpcode::CLASS_OFFSET));

        let fp_f = F::from_canonical_u32(fp);

        // Determine which registers to read based on opcode
        let (pc_source_record, pc_source_data) = match local_opcode {
            JaafOpcode::RET | JaafOpcode::CALL_INDIRECT => {
                // Read pc_source (c field) for target PC
                let pc_source = memory.read::<RV32_REGISTER_NUM_LIMBS>(F::ONE, c + fp_f);
                (Some(pc_source.0), pc_source.1)
            }
            _ => {
                // For JAAF, JAAF_SAVE, and CALL, we don't read pc_source but still need to advance timestamp
                memory.increment_timestamp();
                (None, [F::ZERO; RV32_REGISTER_NUM_LIMBS])
            }
        };

        // All opcodes always read fp_source (e field) for target FP
        let fp_source = memory.read::<RV32_REGISTER_NUM_LIMBS>(F::ONE, e + fp_f);

        Ok((
            [pc_source_data, fp_source.1],
            JaafReadRecord {
                rs1: pc_source_record,
                rs2: fp_source.0,
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        from_frame: FrameState<u32>,
        output: AdapterRuntimeContextWom<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, u32, Self::WriteRecord)> {
        let Instruction {
            a,
            b,
            d: immediate,
            f: enabled, // instruction enable flag - if ZERO, instruction is a no-op
            opcode,
            ..
        } = *instruction;

        let to_fp = output.to_fp.unwrap();

        let local_opcode =
            JaafOpcode::from_usize(opcode.local_opcode_idx(JaafOpcode::CLASS_OFFSET));

        // Determine which registers to write based on opcode
        let (rd1_id, rd2_id) = if enabled != F::ZERO {
            match local_opcode {
                JaafOpcode::JAAF | JaafOpcode::RET => {
                    // No saves
                    memory.increment_timestamp();
                    memory.increment_timestamp();
                    (None, None)
                }
                JaafOpcode::JAAF_SAVE => {
                    // Save fp to rd2 (b field)
                    memory.increment_timestamp();
                    let rd2 =
                        memory.write(F::ONE, b + F::from_canonical_u32(to_fp), output.writes[1]);
                    (None, Some(rd2.0))
                }
                JaafOpcode::CALL | JaafOpcode::CALL_INDIRECT => {
                    // Save both pc to rd1 (a field) and fp to rd2 (b field)
                    let rd1 =
                        memory.write(F::ONE, a + F::from_canonical_u32(to_fp), output.writes[0]);
                    let rd2 =
                        memory.write(F::ONE, b + F::from_canonical_u32(to_fp), output.writes[1]);
                    (Some(rd1.0), Some(rd2.0))
                }
            }
        } else {
            memory.increment_timestamp();
            memory.increment_timestamp();
            (None, None)
        };

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
                timestamp: memory.timestamp(),
            },
            to_fp,
            Self::WriteRecord {
                from_state,
                from_frame,
                rd1_id,
                rd2_id,
                imm: immediate.as_canonical_u32(),
            },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        memory: &OfflineMemory<F>,
    ) {
        let aux_cols_factory = memory.aux_cols_factory();
        let adapter_cols: &mut JaafAdapterColsWom<_> = row_slice.borrow_mut();
        adapter_cols.from_state = write_record.from_state.map(F::from_canonical_u32);
        adapter_cols.from_frame = write_record.from_frame.map(F::from_canonical_u32);
        adapter_cols.imm = F::from_canonical_u32(write_record.imm);

        // Handle rs1 read
        if let Some(rs1_id) = read_record.rs1 {
            let rs1 = memory.record_by_id(rs1_id);
            adapter_cols.rs1_ptr = rs1.pointer;
            adapter_cols.needs_read_rs1 = F::ONE;
            aux_cols_factory.generate_read_aux(rs1, &mut adapter_cols.rs1_aux_cols);
        }

        // Handle rs2 read (always present since FP is always needed)
        let rs2 = memory.record_by_id(read_record.rs2);
        adapter_cols.rs2_ptr = rs2.pointer;
        adapter_cols.needs_read_rs2 = F::ONE;
        aux_cols_factory.generate_read_aux(rs2, &mut adapter_cols.rs2_aux_cols);

        // Handle rd1 write
        if let Some(id) = write_record.rd1_id {
            let rd = memory.record_by_id(id);
            adapter_cols.rd1_ptr = rd.pointer;
            adapter_cols.needs_write_rd1 = F::ONE;
            aux_cols_factory.generate_write_aux(rd, &mut adapter_cols.rd1_aux_cols);
        }

        // Handle rd2 write
        if let Some(id) = write_record.rd2_id {
            let rd = memory.record_by_id(id);
            adapter_cols.rd2_ptr = rd.pointer;
            adapter_cols.needs_write_rd2 = F::ONE;
            aux_cols_factory.generate_write_aux(rd, &mut adapter_cols.rd2_aux_cols);
        }
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

use std::{borrow::Borrow, marker::PhantomData};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, MinimalInstruction, Result, VmAdapterAir, VmAdapterInterface,
    },
    system::{
        memory::{MemoryController, OfflineMemory, offline_checker::MemoryBridge},
        program::ProgramBus,
    },
};
use openvm_circuit_primitives::utils::{compose, select};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::ColumnsAir,
};
use openvm_womir_transpiler::JaafOpcode::{self, *};
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{
    FrameBridge, FrameBus, FrameState, VmAdapterChipWom, WomBridge, WomController, WomRecord,
};

use super::RV32_REGISTER_NUM_LIMBS;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct JaafInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
    pub save_pc: T,
    pub save_fp: T,
    pub read_pc: T,
}

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
        wom_bridge: WomBridge,
    ) -> Self {
        Self {
            air: JaafAdapterAirWom {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                frame_bridge: FrameBridge::new(frame_bus),
                _memory_bridge: memory_bridge,
                wom_bridge,
            },
            _marker: PhantomData,
        }
    }
}
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaafReadRecord<T> {
    pub rs1: Option<WomRecord<T>>,
    pub rs1_data: T,       // PC
    pub rs2: WomRecord<T>, // Always present since FP is always needed
    pub rs2_data: T,       // FP
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaafWriteRecord<T> {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    pub rd1: Option<WomRecord<T>>,
    pub rd2: Option<WomRecord<T>>,
    pub imm: u32,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct JaafAdapterColsWom<T> {
    pub from_state: ExecutionState<T>,
    pub from_frame: FrameState<T>,

    pub pc_read_reg: T,
    // PC stored as a single field element instead of 4 limbs
    pub pc_read_val: T,
    pub pc_save_reg: T,
    // PC stored as a single field element instead of 4 limbs
    pub pc_save_val: T,
    pub pc_write_mult: T,
    pub pc_imm: T,

    pub fp_read_reg: T,
    // FP stored as a single field element instead of 4 limbs
    pub fp_read_val: T,
    pub fp_save_reg: T,
    // FP stored as a single field element instead of 4 limbs
    pub fp_save_val: T,
    pub fp_write_mult: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct JaafAdapterAirWom {
    pub(super) _memory_bridge: MemoryBridge,
    pub(super) wom_bridge: WomBridge,
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
        JaafInstruction<AB::Expr>,
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
        let local: &JaafAdapterColsWom<AB::Var> = local.borrow();

        // read fp
        self.wom_bridge
            .read(
                local.fp_read_reg + local.from_frame.fp,
                [
                    local.fp_read_val.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // read pc
        self.wom_bridge
            .read(
                local.pc_read_reg + local.from_frame.fp,
                [
                    local.pc_read_val.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
            )
            .eval(builder, ctx.instruction.read_pc.clone());

        let to_fp: AB::Expr = compose(&ctx.reads[1], RV32_REGISTER_NUM_LIMBS);

        // save pc
        self.wom_bridge
            .write(
                local.pc_save_reg + to_fp.clone(),
                [
                    local.pc_save_val.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
                local.pc_write_mult,
            )
            .eval(builder, ctx.instruction.save_pc);

        // save fp
        self.wom_bridge
            .write(
                local.fp_save_reg + to_fp.clone(),
                [
                    local.fp_save_val.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
                local.fp_write_mult,
            )
            .eval(builder, ctx.instruction.save_fp);

        let timestamp_change = AB::Expr::ONE;

        let to_pc = select(ctx.instruction.read_pc, local.pc_read_val, local.pc_imm);

        self.execution_bridge
            .execute_and_increment_or_set_pc::<AB>(
                ctx.instruction.opcode,
                [
                    local.pc_save_reg.into(),
                    local.fp_save_reg.into(),
                    local.pc_read_reg.into(),
                    local.pc_imm.into(),
                    local.fp_read_reg.into(),
                    AB::Expr::ONE,
                ],
                local.from_state,
                timestamp_change.clone(),
                (DEFAULT_PC_STEP, Some(to_pc)),
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.frame_bridge
            .set_fp(local.from_frame, timestamp_change, to_fp)
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &JaafAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for JaafAdapterChipWom<F> {
    type ReadRecord = JaafReadRecord<F>;
    type WriteRecord = JaafWriteRecord<F>;
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
        wom: &mut WomController<F>,
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

        memory.increment_timestamp();

        // Determine which registers to read based on opcode
        let (pc_source_record, pc_source_data) = match local_opcode {
            JaafOpcode::RET | JaafOpcode::CALL_INDIRECT => {
                // Read pc_source (c field) for target PC
                let pc_source = wom.read::<RV32_REGISTER_NUM_LIMBS>(c + fp_f);
                (Some(pc_source.0), pc_source.1[0])
            }
            _ => (None, F::ZERO),
        };

        // All opcodes always read fp_source (e field) for target FP
        let fp_source = wom.read::<RV32_REGISTER_NUM_LIMBS>(e + fp_f);

        Ok((
            [
                [pc_source_data, F::ZERO, F::ZERO, F::ZERO],
                [fp_source.1[0], F::ZERO, F::ZERO, F::ZERO],
            ],
            JaafReadRecord {
                rs1: pc_source_record,
                rs1_data: pc_source_data,
                rs2: fp_source.0,
                rs2_data: fp_source.1[0],
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        wom: &mut WomController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        from_frame: FrameState<u32>,
        _output: AdapterRuntimeContext<F, Self::Interface>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, u32, Self::WriteRecord)> {
        let Instruction {
            a,
            b,
            d: immediate,
            f: enabled, // instruction enable flag - if ZERO, instruction is a no-op
            opcode,
            ..
        } = *instruction;

        let local_opcode =
            JaafOpcode::from_usize(opcode.local_opcode_idx(JaafOpcode::CLASS_OFFSET));

        // For RET and CALL_INDIRECT, the immediate should be 0 as PC comes from register
        let imm = match local_opcode {
            RET | CALL_INDIRECT => 0,
            _ => immediate.as_canonical_u32(),
        };

        let pc_source_val = F::as_canonical_u32(&read_record.rs1_data);

        let to_pc = run_jalr(local_opcode, imm, pc_source_val);

        // For all JAAF instructions, we also need to handle fp
        let to_fp = F::as_canonical_u32(&read_record.rs2_data);

        // Prepare writes based on opcode
        let (from_pc, from_fp) = match local_opcode {
            JAAF | RET => {
                // No saves, but we still need to provide write data
                (0, 0)
            }
            JAAF_SAVE => {
                // Save fp to rd2
                (0, from_frame.fp)
            }
            CALL | CALL_INDIRECT => {
                // Save pc to rd1 and fp to rd2
                (from_state.pc, from_frame.fp)
            }
        };

        // Determine which registers to write based on opcode
        let (rd1, rd2) = if enabled != F::ZERO {
            match local_opcode {
                JaafOpcode::JAAF | JaafOpcode::RET => {
                    // No saves
                    (None, None)
                }
                JaafOpcode::JAAF_SAVE => {
                    // Save fp to rd2 (b field)
                    let rd2 = wom.write(
                        b + F::from_canonical_u32(to_fp),
                        [F::from_canonical_u32(from_fp), F::ZERO, F::ZERO, F::ZERO],
                    );
                    (None, Some(rd2))
                }
                JaafOpcode::CALL | JaafOpcode::CALL_INDIRECT => {
                    // Save both pc to rd1 (a field) and fp to rd2 (b field)
                    let rd1 = wom.write(
                        a + F::from_canonical_u32(to_fp),
                        [F::from_canonical_u32(from_pc + DEFAULT_PC_STEP), F::ZERO, F::ZERO, F::ZERO],
                    );
                    let rd2 = wom.write(
                        b + F::from_canonical_u32(to_fp),
                        [F::from_canonical_u32(from_fp), F::ZERO, F::ZERO, F::ZERO],
                    );
                    (Some(rd1), Some(rd2))
                }
            }
        } else {
            (None, None)
        };

        Ok((
            ExecutionState {
                pc: to_pc,
                timestamp: memory.timestamp(),
            },
            to_fp,
            Self::WriteRecord {
                from_state,
                from_frame,
                rd1,
                rd2,
                imm: immediate.as_canonical_u32(),
            },
        ))
    }

    fn generate_trace_row(
        &self,
        _row_slice: &mut [F],
        _read_record: Self::ReadRecord,
        _write_record: Self::WriteRecord,
        _memory: &OfflineMemory<F>,
    ) {
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

pub(super) fn run_jalr(opcode: JaafOpcode, imm: u32, pc_source: u32) -> u32 {
    let to_pc = match opcode {
        JAAF | JAAF_SAVE | CALL => {
            // Use immediate for PC
            imm
        }
        RET | CALL_INDIRECT => {
            // Use pc_source for PC directly (no offset)
            let to_pc = pc_source;
            to_pc - (to_pc & 1)
        }
    };
    assert!(to_pc < (1 << PC_BITS));
    to_pc
}

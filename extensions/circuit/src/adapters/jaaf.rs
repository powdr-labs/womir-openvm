use std::{array, borrow::Borrow, marker::PhantomData};

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
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    riscv::RV32_CELL_BITS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
    rap::ColumnsAir,
};
use openvm_womir_transpiler::JaafOpcode::{self, *};
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{
    FrameBridge, FrameBus, FrameState, VmAdapterChipWom, WomBridge, WomController, WomRecord,
};

use super::{RV32_REGISTER_NUM_LIMBS, compose, decompose};

const RV32_LIMB_MAX: u32 = (1 << RV32_CELL_BITS) - 1;

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
                _execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                _frame_bridge: FrameBridge::new(frame_bus),
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
    pub rs1_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub rs2: WomRecord<T>, // Always present since FP is always needed
    pub rs2_data: [T; RV32_REGISTER_NUM_LIMBS],
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
    pub read_pc_ptr: T,
    pub read_fp_ptr: T,
    pub pc_imm: T,
    pub write_pc_ptr: T,
    pub write_pc_mult: T,
    pub write_fp_ptr: T,
    pub write_fp_mult: T,
    pub needs_save_pc: T,
    pub needs_save_fp: T,
    pub read_pc_imm: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct JaafAdapterAirWom {
    pub(super) _memory_bridge: MemoryBridge,
    pub(super) wom_bridge: WomBridge,
    pub(super) _execution_bridge: ExecutionBridge,
    pub(super) _frame_bridge: FrameBridge,
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
        let local: &JaafAdapterColsWom<AB::Var> = local.borrow();

        builder.assert_bool(local.needs_save_pc);
        builder.assert_bool(local.needs_save_fp);
        builder.assert_bool(local.read_pc_imm);

        self.wom_bridge
            .read(local.read_fp_ptr, ctx.reads[1].clone())
            .eval(builder, ctx.instruction.is_valid.clone());

        // read pc if not immediate
        self.wom_bridge
            .read(local.read_pc_ptr, ctx.reads[0].clone())
            .eval(builder, not(local.read_pc_imm));

        // save pc
        self.wom_bridge
            .write(
                local.write_pc_ptr,
                ctx.writes[0].clone(),
                local.write_pc_mult,
            )
            .eval(builder, local.needs_save_pc);

        // save fp
        self.wom_bridge
            .write(
                local.write_fp_ptr,
                ctx.writes[1].clone(),
                local.write_fp_mult,
            )
            .eval(builder, local.needs_save_fp);
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
                (Some(pc_source.0), pc_source.1)
            }
            _ => {
                // For JAAF, JAAF_SAVE, and CALL, we don't read pc_source but still need to advance timestamp
                (None, [F::ZERO; RV32_REGISTER_NUM_LIMBS])
            }
        };

        // All opcodes always read fp_source (e field) for target FP
        let fp_source = wom.read::<RV32_REGISTER_NUM_LIMBS>(e + fp_f);

        Ok((
            [pc_source_data, fp_source.1],
            JaafReadRecord {
                rs1: pc_source_record,
                rs1_data: pc_source_data,
                rs2: fp_source.0,
                rs2_data: fp_source.1,
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

        let pc_source_data = read_record.rs1_data;
        let pc_source_val = compose(pc_source_data);

        let (to_pc, rd_data) = run_jalr(local_opcode, from_frame.pc, imm, pc_source_val);

        // For all JAAF instructions, we also need to handle fp
        let fp_source_data = read_record.rs2_data;
        let to_fp = compose(fp_source_data);

        let rd_data = rd_data.map(F::from_canonical_u32);

        // Prepare writes based on opcode
        let writes = match local_opcode {
            JAAF | RET => {
                // No saves, but we still need to provide write data
                [
                    [F::ZERO; RV32_REGISTER_NUM_LIMBS],
                    [F::ZERO; RV32_REGISTER_NUM_LIMBS],
                ]
            }
            JAAF_SAVE => {
                // Save fp to rd2
                [
                    [F::ZERO; RV32_REGISTER_NUM_LIMBS],
                    decompose::<F>(from_frame.fp),
                ]
            }
            CALL | CALL_INDIRECT => {
                // Save pc to rd1 and fp to rd2
                [rd_data, decompose::<F>(from_frame.fp)]
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
                    let rd2 = wom.write(b + F::from_canonical_u32(to_fp), writes[1]);
                    (None, Some(rd2))
                }
                JaafOpcode::CALL | JaafOpcode::CALL_INDIRECT => {
                    // Save both pc to rd1 (a field) and fp to rd2 (b field)
                    let rd1 = wom.write(a + F::from_canonical_u32(to_fp), writes[0]);
                    let rd2 = wom.write(b + F::from_canonical_u32(to_fp), writes[1]);
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

pub(super) fn run_jalr(
    opcode: JaafOpcode,
    pc: u32,
    imm: u32,
    pc_source: u32,
) -> (u32, [u32; RV32_REGISTER_NUM_LIMBS]) {
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
    (
        to_pc,
        array::from_fn(|i: usize| ((pc + DEFAULT_PC_STEP) >> (RV32_CELL_BITS * i)) & RV32_LIMB_MAX),
    )
}

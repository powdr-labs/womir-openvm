use std::{borrow::Borrow, marker::PhantomData};

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionBus, ExecutionState,
        MinimalInstruction, Result, VmAdapterAir, VmAdapterInterface,
    },
    system::{
        memory::{offline_checker::MemoryBridge, MemoryController, OfflineMemory, RecordId},
        program::ProgramBus,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
    rap::ColumnsAir,
};
use openvm_womir_transpiler::CopyIntoFrameOpcode;
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{AdapterRuntimeContextWom, FrameBus, FrameState, VmAdapterChipWom};

use super::{compose, decompose, RV32_REGISTER_NUM_LIMBS};

#[derive(Debug)]
pub struct CopyIntoFrameAdapterChipWom<F: Field> {
    pub air: CopyIntoFrameAdapterAirWom,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> CopyIntoFrameAdapterChipWom<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
    ) -> Self {
        Self {
            air: CopyIntoFrameAdapterAirWom {
                _execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                _frame_bus: frame_bus,
                _memory_bridge: memory_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyIntoFrameReadRecord {
    pub rs1: Option<(RecordId, u32)>, // Value to copy
    pub rs2: Option<(RecordId, u32)>, // Frame pointer
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyIntoFrameWriteRecord {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    pub rd: u32,
    pub rd_id: Option<RecordId>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct CopyIntoFrameAdapterColsWom<T> {
    pub from_state: ExecutionState<T>,
    pub from_frame: FrameState<T>,
    pub value_reg_ptr: T, // rs1 pointer (register containing value to copy)
    pub value_reg_aux_cols: [T; 2],
    pub frame_ptr_reg_ptr: T, // rs2 pointer (register containing frame pointer)
    pub frame_ptr_reg_aux_cols: T,
    pub destination_ptr: T, // Where we write: frame_pointer + offset
    /// 0 if copy_into_frame
    /// 1 if copy_from_frame
    pub copy_into_or_from: T,
    pub write_mult: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct CopyIntoFrameAdapterAirWom {
    pub(super) _memory_bridge: MemoryBridge,
    pub(super) _execution_bridge: ExecutionBridge,
    pub(super) _frame_bus: FrameBus,
}

impl<F: Field> BaseAir<F> for CopyIntoFrameAdapterAirWom {
    fn width(&self) -> usize {
        CopyIntoFrameAdapterColsWom::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for CopyIntoFrameAdapterAirWom {
    fn columns(&self) -> Option<Vec<String>> {
        CopyIntoFrameAdapterColsWom::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for CopyIntoFrameAdapterAirWom {
    type Interface = BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 0, 0, 0, 0>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        // Need at least one constraint otherwise stark-backend complains.
        builder.assert_bool(local[0]);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &CopyIntoFrameAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for CopyIntoFrameAdapterChipWom<F> {
    type ReadRecord = CopyIntoFrameReadRecord;
    type WriteRecord = CopyIntoFrameWriteRecord;
    type Air = CopyIntoFrameAdapterAirWom;
    type Interface = BasicAdapterInterface<
        F,
        MinimalInstruction<F>,
        2,
        1,
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
        let Instruction { b, c, opcode, .. } = *instruction;

        // COPY_INTO_FRAME: target_reg (a), src_reg (b), target_fp (c)
        // COPY_FROM_FRAME: target_reg (a), src_reg (b), src_fp (c)

        let fp_f = F::from_canonical_u32(fp);

        let local_opcode = CopyIntoFrameOpcode::from_usize(
            opcode.local_opcode_idx(CopyIntoFrameOpcode::CLASS_OFFSET),
        );

        let other_fp = memory.read::<RV32_REGISTER_NUM_LIMBS>(F::ONE, c + fp_f);

        let other_fp_u32 = compose(other_fp.1);
        let other_fp_f = F::from_canonical_u32(other_fp_u32);

        let value_to_copy = match local_opcode {
            CopyIntoFrameOpcode::COPY_INTO_FRAME => {
                memory.read::<RV32_REGISTER_NUM_LIMBS>(F::ONE, b + fp_f)
            }
            CopyIntoFrameOpcode::COPY_FROM_FRAME => {
                memory.read::<RV32_REGISTER_NUM_LIMBS>(F::ONE, b + other_fp_f)
            }
        };

        Ok((
            [value_to_copy.1, other_fp.1],
            CopyIntoFrameReadRecord {
                rs1: Some((value_to_copy.0, compose(value_to_copy.1))),
                rs2: Some((other_fp.0, compose(other_fp.1))),
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        from_frame: FrameState<u32>,
        _output: AdapterRuntimeContextWom<F, Self::Interface>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, u32, Self::WriteRecord)> {
        let Instruction {
            a,
            f: enabled,
            opcode,
            ..
        } = *instruction;

        let mut destination_id = None;

        let local_opcode = CopyIntoFrameOpcode::from_usize(
            opcode.local_opcode_idx(CopyIntoFrameOpcode::CLASS_OFFSET),
        );

        let fp_f = F::from_canonical_u32(from_frame.fp);

        if enabled != F::ZERO {
            let value = read_record.rs1.unwrap().1;
            let other_fp = read_record.rs2.unwrap().1;
            let other_fp_f = F::from_canonical_u32(other_fp);

            let write_result = match local_opcode {
                CopyIntoFrameOpcode::COPY_INTO_FRAME => {
                    memory.write(F::ONE, a + other_fp_f, decompose(value))
                }
                CopyIntoFrameOpcode::COPY_FROM_FRAME => {
                    memory.write(F::ONE, a + fp_f, decompose(value))
                }
            };
            destination_id = Some(write_result.0);
        }

        Ok((
            ExecutionState {
                pc: from_state.pc + DEFAULT_PC_STEP,
                timestamp: memory.timestamp(),
            },
            from_frame.fp,
            Self::WriteRecord {
                from_state,
                from_frame,
                rd: a.as_canonical_u32(),
                rd_id: destination_id,
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

use std::{borrow::Borrow, marker::PhantomData};

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionBus, ExecutionState,
        MinimalInstruction, Result, VmAdapterAir, VmAdapterInterface,
    },
    system::{
        memory::{MemoryController, OfflineMemory, RecordId, offline_checker::MemoryBridge},
        program::ProgramBus,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
    rap::ColumnsAir,
};
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{AdapterRuntimeContextWom, FrameBus, FrameState, VmAdapterChipWom};

use super::{RV32_REGISTER_NUM_LIMBS, decompose};

#[derive(Debug)]
pub struct ConstsAdapterChipWom<F: Field> {
    pub air: ConstsAdapterAirWom,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> ConstsAdapterChipWom<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
    ) -> Self {
        Self {
            air: ConstsAdapterAirWom {
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
pub struct ConstsWriteRecord {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    pub rd: u32,
    pub rd_id: Option<RecordId>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct ConstsAdapterColsWom<T> {
    pub from_state: ExecutionState<T>,
    pub from_frame: FrameState<T>,
    pub value_reg_ptr: T,
    pub lo: T,
    pub hi: T,
    pub write_mult: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct ConstsAdapterAirWom {
    pub(super) _memory_bridge: MemoryBridge,
    pub(super) _execution_bridge: ExecutionBridge,
    pub(super) _frame_bus: FrameBus,
}

impl<F: Field> BaseAir<F> for ConstsAdapterAirWom {
    fn width(&self) -> usize {
        ConstsAdapterColsWom::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for ConstsAdapterAirWom {
    fn columns(&self) -> Option<Vec<String>> {
        ConstsAdapterColsWom::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for ConstsAdapterAirWom {
    type Interface = BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 0, 0, 0, 0>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        builder.assert_bool(local[0]);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &ConstsAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for ConstsAdapterChipWom<F> {
    type ReadRecord = ();
    type WriteRecord = ConstsWriteRecord;
    type Air = ConstsAdapterAirWom;
    type Interface = BasicAdapterInterface<
        F,
        MinimalInstruction<F>,
        0,
        1,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn preprocess(
        &mut self,
        _memory: &mut MemoryController<F>,
        _fp: u32,
        _instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        Ok(([], ()))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        from_frame: FrameState<u32>,
        _output: AdapterRuntimeContextWom<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, u32, Self::WriteRecord)> {
        let Instruction {
            a,
            b,
            c,
            f: enabled,
            ..
        } = *instruction;

        let mut destination_id = None;

        if enabled != F::ZERO {
            let imm_lo = b.as_canonical_u32();
            let imm_hi = c.as_canonical_u32();
            assert!(
                imm_lo < (1 << 16) && imm_hi < (1 << 16),
                "Immediate values out of range",
            );
            let imm = imm_hi << 16 | imm_lo;
            let fp_f = F::from_canonical_u32(from_frame.fp);
            let write_result = memory.write(F::ONE, a + fp_f, decompose(imm));
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

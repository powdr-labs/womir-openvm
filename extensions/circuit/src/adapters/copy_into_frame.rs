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
            MemoryController, OfflineMemory, RecordId,
        },
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
    pub offset_within_frame: T, // rd - the offset within the frame
    pub value_reg_ptr: T,       // rs1 pointer (register containing value to copy)
    pub value_reg_aux_cols: MemoryReadAuxCols<T>,
    pub frame_ptr_reg_ptr: T, // rs2 pointer (register containing frame pointer)
    pub frame_ptr_reg_aux_cols: MemoryReadAuxCols<T>,
    pub destination_ptr: T, // Where we write: frame_pointer + offset
    pub destination_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    /// 1 if we need to write to destination
    pub needs_write: T,
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
        _builder: &mut AB,
        _local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        // Empty eval function as requested
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
        let Instruction { b, c, .. } = *instruction;

        // COPY_INTO_FRAME: target_reg (a), src_reg (b), target_fp (c)
        let fp_f = F::from_canonical_u32(fp);
        let value_to_copy = memory.read::<RV32_REGISTER_NUM_LIMBS>(F::ONE, b + fp_f);
        let future_fp = memory.read::<RV32_REGISTER_NUM_LIMBS>(F::ONE, c + fp_f);

        Ok((
            [value_to_copy.1, future_fp.1],
            CopyIntoFrameReadRecord {
                rs1: Some((value_to_copy.0, compose(value_to_copy.1))),
                rs2: Some((future_fp.0, compose(future_fp.1))),
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
        let Instruction { a, f: enabled, .. } = *instruction;

        let mut destination_id = None;

        if enabled != F::ZERO {
            let value = read_record.rs1.unwrap().1;
            let future_fp = read_record.rs2.unwrap().1;
            let future_fp_f = F::from_canonical_u32(future_fp);
            let write_result = memory.write(F::ONE, a + future_fp_f, decompose(value));
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
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        memory: &OfflineMemory<F>,
    ) {
        let aux_cols_factory = memory.aux_cols_factory();
        let adapter_cols: &mut CopyIntoFrameAdapterColsWom<_> = row_slice.borrow_mut();

        adapter_cols.from_state = write_record.from_state.map(F::from_canonical_u32);
        adapter_cols.from_frame = write_record.from_frame.map(F::from_canonical_u32);
        adapter_cols.offset_within_frame = F::from_canonical_u32(write_record.rd);

        // Handle value register read (rs1)
        if let Some(value_id) = read_record.rs1 {
            let value_record = memory.record_by_id(value_id.0);
            adapter_cols.value_reg_ptr = value_record.pointer;
            aux_cols_factory.generate_read_aux(value_record, &mut adapter_cols.value_reg_aux_cols);
        }

        // Handle frame pointer register read (rs2)
        if let Some(frame_ptr_id) = read_record.rs2 {
            let frame_ptr_record = memory.record_by_id(frame_ptr_id.0);
            adapter_cols.frame_ptr_reg_ptr = frame_ptr_record.pointer;
            aux_cols_factory
                .generate_read_aux(frame_ptr_record, &mut adapter_cols.frame_ptr_reg_aux_cols);
        }

        // Handle destination write
        if let Some(dest_id) = write_record.rd_id {
            let dest_record = memory.record_by_id(dest_id);
            adapter_cols.destination_ptr = dest_record.pointer;
            adapter_cols.needs_write = F::ONE;
            aux_cols_factory
                .generate_write_aux(dest_record, &mut adapter_cols.destination_aux_cols);
        } else {
            adapter_cols.needs_write = F::ZERO;
        }
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

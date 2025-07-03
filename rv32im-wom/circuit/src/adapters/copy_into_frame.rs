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

use super::RV32_REGISTER_NUM_LIMBS;

#[derive(Debug)]
pub struct Rv32CopyIntoFrameAdapterChipWom<F: Field> {
    pub air: Rv32CopyIntoFrameAdapterAirWom,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> Rv32CopyIntoFrameAdapterChipWom<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
    ) -> Self {
        Self {
            air: Rv32CopyIntoFrameAdapterAirWom {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                frame_bus,
                memory_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rv32CopyIntoFrameReadRecord {
    pub rs1: Option<RecordId>, // Value to copy
    pub rs2: Option<RecordId>, // Frame pointer
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rv32CopyIntoFrameWriteRecord {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    pub rd: u32,
    pub rd_id: Option<RecordId>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv32CopyIntoFrameAdapterColsWom<T> {
    pub from_state: ExecutionState<T>,
    pub from_frame: FrameState<T>,
    pub rd: T,
    pub rs1_ptr: T,
    pub rs1_aux_cols: MemoryReadAuxCols<T>,
    pub rs2_ptr: T,
    pub rs2_aux_cols: MemoryReadAuxCols<T>,
    pub rd_ptr: T,
    pub rd_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    /// 1 if we need to write rd
    pub needs_write_rd: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32CopyIntoFrameAdapterAirWom {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) frame_bus: FrameBus,
}

impl<F: Field> BaseAir<F> for Rv32CopyIntoFrameAdapterAirWom {
    fn width(&self) -> usize {
        Rv32CopyIntoFrameAdapterColsWom::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for Rv32CopyIntoFrameAdapterAirWom {
    fn columns(&self) -> Option<Vec<String>> {
        Rv32CopyIntoFrameAdapterColsWom::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32CopyIntoFrameAdapterAirWom {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        1,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        // Empty eval function as requested
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32CopyIntoFrameAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for Rv32CopyIntoFrameAdapterChipWom<F> {
    type ReadRecord = Rv32CopyIntoFrameReadRecord;
    type WriteRecord = Rv32CopyIntoFrameWriteRecord;
    type Air = Rv32CopyIntoFrameAdapterAirWom;
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
        _fp: u32,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { a, b, .. } = *instruction;

        // COPY_INTO_FRAME: rs2 (a), rs1 (b), rd (c)
        // Read rs1 (value to copy) and rs2 (frame pointer)
        let rs1 = memory.read::<RV32_REGISTER_NUM_LIMBS>(F::ONE, b);
        let rs2 = memory.read::<RV32_REGISTER_NUM_LIMBS>(F::ONE, a);

        Ok((
            [rs1.1, rs2.1],
            Rv32CopyIntoFrameReadRecord {
                rs1: Some(rs1.0),
                rs2: Some(rs2.0),
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
        let Instruction { c, f: enabled, .. } = *instruction;

        let mut rd_id = None;

        if enabled != F::ZERO {
            // Write the result to rd register
            if let Some(writes) = output.writes.first() {
                let write_result = memory.write(F::ONE, c, *writes);
                rd_id = Some(write_result.0);
            }
        }

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
                timestamp: memory.timestamp(),
            },
            output.to_fp.unwrap_or(from_frame.fp),
            Self::WriteRecord {
                from_state,
                from_frame,
                rd: c.as_canonical_u32(),
                rd_id,
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
        let adapter_cols: &mut Rv32CopyIntoFrameAdapterColsWom<_> = row_slice.borrow_mut();

        adapter_cols.from_state = write_record.from_state.map(F::from_canonical_u32);
        adapter_cols.from_frame = write_record.from_frame.map(F::from_canonical_u32);
        adapter_cols.rd = F::from_canonical_u32(write_record.rd);

        // Handle rs1 read
        if let Some(rs1_id) = read_record.rs1 {
            let rs1_record = memory.record_by_id(rs1_id);
            adapter_cols.rs1_ptr = rs1_record.pointer;
            aux_cols_factory.generate_read_aux(rs1_record, &mut adapter_cols.rs1_aux_cols);
        }

        // Handle rs2 read
        if let Some(rs2_id) = read_record.rs2 {
            let rs2_record = memory.record_by_id(rs2_id);
            adapter_cols.rs2_ptr = rs2_record.pointer;
            aux_cols_factory.generate_read_aux(rs2_record, &mut adapter_cols.rs2_aux_cols);
        }

        // Handle rd write
        if let Some(rd_id) = write_record.rd_id {
            let rd_record = memory.record_by_id(rd_id);
            adapter_cols.rd_ptr = rd_record.pointer;
            adapter_cols.needs_write_rd = F::ONE;
            aux_cols_factory.generate_write_aux(rd_record, &mut adapter_cols.rd_aux_cols);
        } else {
            adapter_cols.needs_write_rd = F::ZERO;
        }
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

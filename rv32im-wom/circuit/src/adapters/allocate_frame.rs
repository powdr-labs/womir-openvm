use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionBus, ExecutionState,
        MinimalInstruction, Result, VmAdapterAir, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryWriteAuxCols},
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

use crate::{
    adapters::decompose, AdapterRuntimeContextWom, FrameBus, FrameState, VmAdapterChipWom,
};

use super::RV32_REGISTER_NUM_LIMBS;

#[derive(Debug)]
pub struct Rv32AllocateFrameAdapterChipWom {
    pub air: Rv32AllocateFrameAdapterAirWom,
    next_fp: u32,
}

impl Rv32AllocateFrameAdapterChipWom {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
    ) -> Self {
        Self {
            air: Rv32AllocateFrameAdapterAirWom {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                frame_bus,
                memory_bridge,
            },
            // Start from 1 because 0 is used by the startup code.
            next_fp: 1,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rv32AllocateFrameReadRecord {
    // No reads needed for allocate_frame
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rv32AllocateFrameWriteRecord {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    pub target_reg: u32,
    pub amount_imm: u32,
    pub rd_id: Option<RecordId>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv32AllocateFrameAdapterColsWom<T> {
    pub from_state: ExecutionState<T>,
    pub from_frame: FrameState<T>,
    pub target_reg_offset: T, // target_reg field from instruction (a)
    pub allocation_size: T,   // amount_imm field from instruction (b)
    pub target_reg_ptr: T,    // Pointer to target register
    pub target_reg_aux_cols: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    /// 1 if we need to write to target register
    pub needs_write: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32AllocateFrameAdapterAirWom {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) frame_bus: FrameBus,
}

impl<F: Field> BaseAir<F> for Rv32AllocateFrameAdapterAirWom {
    fn width(&self) -> usize {
        Rv32AllocateFrameAdapterColsWom::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for Rv32AllocateFrameAdapterAirWom {
    fn columns(&self) -> Option<Vec<String>> {
        Rv32AllocateFrameAdapterColsWom::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32AllocateFrameAdapterAirWom {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        0,
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
        let cols: &Rv32AllocateFrameAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for Rv32AllocateFrameAdapterChipWom {
    type ReadRecord = Rv32AllocateFrameReadRecord;
    type WriteRecord = Rv32AllocateFrameWriteRecord;
    type Air = Rv32AllocateFrameAdapterAirWom;
    type Interface = BasicAdapterInterface<
        F,
        MinimalInstruction<F>,
        1,
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
        let Instruction { b, .. } = *instruction;

        memory.increment_timestamp();

        let allocated_ptr = decompose(self.next_fp);
        self.next_fp += b.as_canonical_u32();

        Ok(([allocated_ptr], Rv32AllocateFrameReadRecord {}))
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
            a, b, f: enabled, ..
        } = *instruction;

        let mut target_reg_id = None;

        if enabled != F::ZERO {
            // Write the allocated pointer to target register
            // For simplicity in the mock, we use absolute addressing for the write
            if let Some(allocated_ptr) = output.writes.first() {
                let write_result = memory.write(
                    F::ONE,
                    a + F::from_canonical_u32(from_frame.fp),
                    *allocated_ptr,
                );
                target_reg_id = Some(write_result.0);
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
                target_reg: a.as_canonical_u32(),
                amount_imm: b.as_canonical_u32(),
                rd_id: target_reg_id,
            },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        _read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        memory: &OfflineMemory<F>,
    ) {
        let aux_cols_factory = memory.aux_cols_factory();
        let adapter_cols: &mut Rv32AllocateFrameAdapterColsWom<_> = row_slice.borrow_mut();

        adapter_cols.from_state = write_record.from_state.map(F::from_canonical_u32);
        adapter_cols.from_frame = write_record.from_frame.map(F::from_canonical_u32);
        adapter_cols.target_reg_offset = F::from_canonical_u32(write_record.target_reg);
        adapter_cols.allocation_size = F::from_canonical_u32(write_record.amount_imm);

        // Handle target register write
        if let Some(target_id) = write_record.rd_id {
            let target_record = memory.record_by_id(target_id);
            adapter_cols.target_reg_ptr = target_record.pointer;
            adapter_cols.needs_write = F::ONE;
            aux_cols_factory
                .generate_write_aux(target_record, &mut adapter_cols.target_reg_aux_cols);
        } else {
            adapter_cols.needs_write = F::ZERO;
        }
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

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
    adapters::{compose, decompose},
    AdapterRuntimeContextWom, FrameBus, FrameState, VmAdapterChipWom,
};

use super::RV32_REGISTER_NUM_LIMBS;

#[derive(Debug)]
pub struct AllocateFrameAdapterChipWom {
    pub air: AllocateFrameAdapterAirWom,
    next_fp: u32,
}

impl AllocateFrameAdapterChipWom {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        frame_bus: FrameBus,
        memory_bridge: MemoryBridge,
    ) -> Self {
        Self {
            air: AllocateFrameAdapterAirWom {
                _execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                _frame_bus: frame_bus,
                _memory_bridge: memory_bridge,
            },
            // Start from 8 because 0 and 4 are used by the startup code.
            next_fp: 8,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocateFrameReadRecord {
    // No reads needed for allocate_frame
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocateFrameWriteRecord {
    pub from_state: ExecutionState<u32>,
    pub from_frame: FrameState<u32>,
    pub target_reg: u32,
    pub amount_imm: u32,
    pub rd_id: Option<RecordId>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct AllocateFrameAdapterColsWom<T> {
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
pub struct AllocateFrameAdapterAirWom {
    pub(super) _memory_bridge: MemoryBridge,
    pub(super) _execution_bridge: ExecutionBridge,
    pub(super) _frame_bus: FrameBus,
}

impl<F: Field> BaseAir<F> for AllocateFrameAdapterAirWom {
    fn width(&self) -> usize {
        AllocateFrameAdapterColsWom::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for AllocateFrameAdapterAirWom {
    fn columns(&self) -> Option<Vec<String>> {
        AllocateFrameAdapterColsWom::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for AllocateFrameAdapterAirWom {
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
        builder: &mut AB,
        local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        // Need at least one constraint otherwise stark-backend complains.
        builder.assert_bool(local[0]);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &AllocateFrameAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for AllocateFrameAdapterChipWom {
    type ReadRecord = AllocateFrameReadRecord;
    type WriteRecord = AllocateFrameWriteRecord;
    type Air = AllocateFrameAdapterAirWom;
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
        let Instruction {
            b: amount_imm,
            c: amount_reg,
            d: use_reg,
            ..
        } = *instruction;

        let amount = if use_reg == F::ZERO {
            // If use_reg is zero, we use the immediate value
            memory.increment_timestamp();
            amount_imm.as_canonical_u32()
        } else {
            // Otherwise, we read the value from the register
            let fp_f = F::from_canonical_u32(fp);
            let reg_value = memory.read::<RV32_REGISTER_NUM_LIMBS>(F::ONE, amount_reg + fp_f);
            compose(reg_value.1)
        };
        let amount_bytes = RV32_REGISTER_NUM_LIMBS as u32 * amount;

        let allocated_ptr = decompose(self.next_fp);
        self.next_fp += amount_bytes;
        let amount_bytes = decompose(amount_bytes);

        Ok(([allocated_ptr, amount_bytes], AllocateFrameReadRecord {}))
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
        let adapter_cols: &mut AllocateFrameAdapterColsWom<_> = row_slice.borrow_mut();

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

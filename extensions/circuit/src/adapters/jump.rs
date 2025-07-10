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
            offline_checker::{MemoryBridge, MemoryReadAuxCols},
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
use openvm_womir_transpiler::JumpOpcode;
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{AdapterRuntimeContextWom, VmAdapterChipWom};

use super::RV32_REGISTER_NUM_LIMBS;

#[derive(Debug)]
pub struct JumpAdapterChipWom<F: Field> {
    pub air: JumpAdapterAirWom,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> JumpAdapterChipWom<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
    ) -> Self {
        Self {
            air: JumpAdapterAirWom {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JumpReadRecord {
    pub condition: Option<RecordId>, // condition register - only for JUMP_IF and JUMP_IF_ZERO
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JumpWriteRecord {
    pub from_state: ExecutionState<u32>,
    pub immediate: u32,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct JumpAdapterColsWom<T> {
    pub from_state: ExecutionState<T>,
    pub condition_ptr: T,
    pub condition_aux_cols: MemoryReadAuxCols<T>,
    /// 1 if we need to read condition (for JUMP_IF and JUMP_IF_ZERO)
    pub needs_read_condition: T,
    /// Immediate value from instruction
    pub immediate: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct JumpAdapterAirWom {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

impl<F: Field> BaseAir<F> for JumpAdapterAirWom {
    fn width(&self) -> usize {
        JumpAdapterColsWom::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for JumpAdapterAirWom {
    fn columns(&self) -> Option<Vec<String>> {
        JumpAdapterColsWom::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for JumpAdapterAirWom {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        1,
        0,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &JumpAdapterColsWom<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let read_count_condition = local_cols.needs_read_condition;

        builder.assert_bool(read_count_condition);
        builder
            .when::<AB::Expr>(not(ctx.instruction.is_valid.clone()))
            .assert_zero(read_count_condition);

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.condition_ptr,
                ),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local_cols.condition_aux_cols,
            )
            .eval(builder, read_count_condition);

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP));

        // Execute the instruction
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.immediate.into(),     // a: immediate
                    local_cols.condition_ptr.into(), // b: condition register
                    AB::Expr::ZERO,                  // c: (not used)
                    AB::Expr::ZERO,                  // d: (not used)
                    AB::Expr::ZERO,                  // e: (not used)
                    AB::Expr::ONE,                   // f: enabled
                    AB::Expr::ZERO,                  // g: imm sign
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
        let cols: &JumpAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for JumpAdapterChipWom<F> {
    type ReadRecord = JumpReadRecord;
    type WriteRecord = JumpWriteRecord;
    type Air = JumpAdapterAirWom;
    type Interface = BasicAdapterInterface<
        F,
        MinimalInstruction<F>,
        1,
        0,
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
        let Instruction {
            a: immediate,
            b,
            opcode,
            ..
        } = *instruction;

        let local_opcode =
            JumpOpcode::from_usize(opcode.local_opcode_idx(JumpOpcode::CLASS_OFFSET));

        // Determine which registers to read based on opcode
        let (condition_record, condition_data) = match local_opcode {
            JumpOpcode::JUMP_IF | JumpOpcode::JUMP_IF_ZERO => {
                // Read condition (b field) for conditional jumps
                let condition = memory.read::<RV32_REGISTER_NUM_LIMBS>(F::ONE, b);
                (Some(condition.0), condition.1)
            }
            _ => {
                // For JUMP, we don't read condition but still need to advance timestamp
                memory.increment_timestamp();
                (None, [F::ZERO; RV32_REGISTER_NUM_LIMBS])
            }
        };

        Ok((
            [condition_data],
            JumpReadRecord {
                condition: condition_record,
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        _from_frame: crate::FrameState<u32>,
        output: AdapterRuntimeContextWom<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, u32, Self::WriteRecord)> {
        let Instruction {
            a: immediate,
            f: enabled,
            ..
        } = *instruction;

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
                timestamp: memory.timestamp(),
            },
            _from_frame.fp, // FP unchanged for jump instructions
            Self::WriteRecord {
                from_state,
                immediate: immediate.as_canonical_u32(),
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
        let adapter_cols: &mut JumpAdapterColsWom<_> = row_slice.borrow_mut();
        adapter_cols.from_state = write_record.from_state.map(F::from_canonical_u32);
        adapter_cols.immediate = F::from_canonical_u32(write_record.immediate);

        // Handle condition read
        if let Some(condition_id) = read_record.condition {
            let condition = memory.record_by_id(condition_id);
            adapter_cols.condition_ptr = condition.pointer;
            adapter_cols.needs_read_condition = F::ONE;
            aux_cols_factory.generate_read_aux(condition, &mut adapter_cols.condition_aux_cols);
        }
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

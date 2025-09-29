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
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
    rap::ColumnsAir,
};
use openvm_womir_transpiler::JumpOpcode::{self, *};
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{VmAdapterChipWom, WomBridge, WomController, WomRecord};

use super::{RV32_REGISTER_NUM_LIMBS, compose};

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
        wom_bridge: WomBridge,
    ) -> Self {
        Self {
            air: JumpAdapterAirWom {
                _execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                _memory_bridge: memory_bridge,
                _wom_bridge: wom_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JumpReadRecord<T> {
    pub reg_value: Option<WomRecord<T>>, // condition register for JUMP_IF and JUMP_IF_ZERO, offset for SKIP
    pub reg_data: [T; 4],
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
    pub condition_or_skip_offset_ptr: T,
    pub condition_or_skip_offset_aux_cols: T,
    pub immediate: T,
    pub opcode_jump: T,
    pub opcode_jump_if_zero: T,
    pub opcode_jump_if: T,
    pub opcode_skip: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct JumpAdapterAirWom {
    pub(super) _memory_bridge: MemoryBridge,
    pub(super) _wom_bridge: WomBridge,
    pub(super) _execution_bridge: ExecutionBridge,
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
        0,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        _ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &JumpAdapterColsWom<AB::Var> = local.borrow();
        let opcode_skip = local_cols.opcode_skip;

        builder.assert_bool(opcode_skip);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &JumpAdapterColsWom<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChipWom<F> for JumpAdapterChipWom<F> {
    type ReadRecord = JumpReadRecord<F>;
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
        wom: &mut WomController<F>,
        fp: u32,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { b, opcode, .. } = *instruction;

        let local_opcode =
            JumpOpcode::from_usize(opcode.local_opcode_idx(JumpOpcode::CLASS_OFFSET));

        let fp_f = F::from_canonical_u32(fp);

        memory.increment_timestamp();

        // Determine which registers to read based on opcode
        let (reg_value, reg_data) = match local_opcode {
            JumpOpcode::JUMP_IF | JumpOpcode::JUMP_IF_ZERO | JumpOpcode::SKIP => {
                // Read condition (b field) for conditional jumps, or the offset for skip
                let reg_value = wom.read::<RV32_REGISTER_NUM_LIMBS>(b + fp_f);
                (Some(reg_value.0), reg_value.1)
            }
            _ => {
                // For JUMP, we don't read condition
                (None, [F::ZERO; RV32_REGISTER_NUM_LIMBS])
            }
        };

        Ok((
            [reg_data],
            JumpReadRecord {
                reg_value,
                reg_data,
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        _wom: &mut WomController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        _from_frame: crate::FrameState<u32>,
        _output: AdapterRuntimeContext<F, Self::Interface>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, u32, Self::WriteRecord)> {
        let Instruction {
            opcode,
            a: immediate,
            ..
        } = *instruction;

        let local_opcode =
            JumpOpcode::from_usize(opcode.local_opcode_idx(JumpOpcode::CLASS_OFFSET));

        let register_val = compose(read_record.reg_data);

        let target_pc = if let SKIP = local_opcode {
            // Skip register_val instructions
            from_state.pc + (register_val + 1) * DEFAULT_PC_STEP
        } else {
            // Jump directly to immediate value
            immediate.as_canonical_u32()
        };

        // Determine if we should jump based on opcode and condition
        let should_jump = match local_opcode {
            JUMP | SKIP => true,               // Always jump
            JUMP_IF => register_val != 0,      // Jump if condition != 0
            JUMP_IF_ZERO => register_val == 0, // Jump if condition == 0
        };

        let final_pc = if should_jump {
            target_pc
        } else {
            from_state.pc + DEFAULT_PC_STEP
        };

        Ok((
            ExecutionState {
                pc: final_pc,
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

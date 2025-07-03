use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::arch::{
    AdapterAirContext, MinimalInstruction, Result, VmAdapterInterface, VmCoreAir,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode,
};
use openvm_rv32im_wom_transpiler::Rv32JumpOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{AdapterRuntimeContextWom, VmCoreChipWom};

use crate::adapters::{compose, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

const RV32_LIMB_MAX: u32 = (1 << RV32_CELL_BITS) - 1;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv32JumpCoreCols<T> {
    pub imm: T,
    pub condition_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub is_valid: T,
    pub should_jump: T, // 1 if jump should happen, 0 if not

    pub to_pc_least_sig_bit: T,
    /// These are the limbs of `to_pc * 2`.
    pub to_pc_limbs: [T; 2],
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct Rv32JumpCoreRecord<F> {
    pub imm: F,
    pub condition_data: [F; RV32_REGISTER_NUM_LIMBS],
    pub should_jump: F,
    pub to_pc_least_sig_bit: F,
    pub to_pc_limbs: [u32; 2],
}

#[derive(Debug, Clone)]
pub struct Rv32JumpCoreAir {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv32JumpCoreAir {
    fn width(&self) -> usize {
        Rv32JumpCoreCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for Rv32JumpCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        Rv32JumpCoreCols::<F>::struct_reflection()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32JumpCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv32JumpCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 0]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv32JumpCoreCols<AB::Var> = (*local_core).borrow();
        let Rv32JumpCoreCols::<AB::Var> {
            imm,
            condition_data,
            is_valid,
            should_jump,
            to_pc_least_sig_bit,
            to_pc_limbs,
        } = *cols;

        builder.assert_bool(is_valid);
        builder.assert_bool(should_jump);

        // Constrain to_pc_least_sig_bit + 2 * to_pc_limbs = imm as a u32 addition
        let inv = AB::F::from_canonical_u32(1 << 16).inverse();

        builder.assert_bool(to_pc_least_sig_bit);
        let carry = (imm - to_pc_limbs[0] * AB::F::TWO - to_pc_least_sig_bit) * inv;
        builder.when(is_valid).assert_bool(carry.clone());

        // No sign extension needed since immediates are always positive
        let carry = (carry - to_pc_limbs[1]) * inv;
        builder.when(is_valid).assert_bool(carry);

        // preventing to_pc overflow
        self.range_bus
            .range_check(to_pc_limbs[1], PC_BITS - 16)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(to_pc_limbs[0], 15)
            .eval(builder, is_valid);
        let to_pc =
            to_pc_limbs[0] * AB::F::TWO + to_pc_limbs[1] * AB::F::from_canonical_u32(1 << 16);

        // Calculate final PC: if should_jump, use to_pc, otherwise use from_pc + DEFAULT_PC_STEP
        let final_pc = should_jump * to_pc.clone() + (AB::Expr::ONE - should_jump) * (from_pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP));

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, JUMP);

        AdapterAirContext {
            to_pc: Some(final_pc),
            reads: [condition_data.map(|x| x.into())].into(), // condition register for conditional jumps
            writes: [].into(), // No writes for jump instructions
            instruction: MinimalInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv32JumpOpcode::CLASS_OFFSET
    }
}

pub struct Rv32JumpCoreChipWom {
    pub air: Rv32JumpCoreAir,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl Rv32JumpCoreChipWom {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        assert!(range_checker_chip.range_max_bits() >= 16);
        Self {
            air: Rv32JumpCoreAir {
                bitwise_lookup_bus: bitwise_lookup_chip.bus(),
                range_bus: range_checker_chip.bus(),
            },
            bitwise_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChipWom<F, I> for Rv32JumpCoreChipWom
where
    I::Reads: Into<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 0]>,
{
    type Record = Rv32JumpCoreRecord<F>;
    type Air = Rv32JumpCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        _from_fp: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContextWom<F, I>, Self::Record)> {
        let Instruction { opcode, a, .. } = *instruction;
        let local_opcode =
            Rv32JumpOpcode::from_usize(opcode.local_opcode_idx(Rv32JumpOpcode::CLASS_OFFSET));

        let imm = a.as_canonical_u32();
        let target_pc = imm; // Jump directly to immediate value

        let reads_array: [[F; RV32_REGISTER_NUM_LIMBS]; 1] = reads.into();
        let condition_data = reads_array[0];
        let condition_val = compose(condition_data);

        // Determine if we should jump based on opcode and condition
        let should_jump = match local_opcode {
            JUMP => true, // Always jump
            JUMP_IF => condition_val != 0, // Jump if condition != 0
            JUMP_IF_ZERO => condition_val == 0, // Jump if condition == 0
        };

        let final_pc = if should_jump {
            target_pc
        } else {
            from_pc + DEFAULT_PC_STEP
        };

        let mask = (1 << 15) - 1;
        let to_pc_least_sig_bit = imm & 1;
        let to_pc_limbs = array::from_fn(|i| ((target_pc >> (1 + i * 15)) & mask));

        self.range_checker_chip
            .add_count(to_pc_limbs[0], 15);
        self.range_checker_chip
            .add_count(to_pc_limbs[1], PC_BITS - 16);

        let output = AdapterRuntimeContextWom {
            to_pc: Some(final_pc),
            to_fp: None, // Jump instructions don't modify FP
            writes: [].into(), // No writes
        };

        Ok((
            output,
            Rv32JumpCoreRecord {
                imm: a,
                condition_data,
                should_jump: F::from_canonical_u32(if should_jump { 1 } else { 0 }),
                to_pc_least_sig_bit: F::from_canonical_u32(to_pc_least_sig_bit),
                to_pc_limbs,
            },
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32JumpOpcode::from_usize(opcode - Rv32JumpOpcode::CLASS_OFFSET)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let core_cols: &mut Rv32JumpCoreCols<F> = row_slice.borrow_mut();
        core_cols.imm = record.imm;
        core_cols.condition_data = record.condition_data;
        core_cols.should_jump = record.should_jump;
        core_cols.to_pc_least_sig_bit = record.to_pc_least_sig_bit;
        core_cols.to_pc_limbs = record.to_pc_limbs.map(F::from_canonical_u32);
        core_cols.is_valid = F::ONE;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
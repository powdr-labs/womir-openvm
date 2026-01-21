use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::{
    AdapterAirContext, VmAdapterInterface, VmCoreAir,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{LocalOpcode, instruction::Instruction};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use openvm_womir_transpiler::JumpOpcode;
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{adapters::{JumpInstruction, RV32_REGISTER_NUM_LIMBS}, VmCoreChipWom, AdapterRuntimeContextWom};

use strum::IntoEnumIterator;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct JumpCoreCols<T> {
    pub opcode_jump_flag: T,
    pub opcode_jump_if_zero_flag: T,
    pub opcode_jump_if_flag: T,
    pub opcode_skip_flag: T,
}

#[derive(Default, Debug, Clone)]
pub struct JumpCoreAir {}

impl<F: Field> BaseAir<F> for JumpCoreAir {
    fn width(&self) -> usize {
        JumpCoreCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for JumpCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        JumpCoreCols::<F>::struct_reflection()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for JumpCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for JumpCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; 0]; 0]>,
    I::ProcessedInstruction: From<JumpInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &JumpCoreCols<AB::Var> = (*local_core).borrow();
        let flags = [
            cols.opcode_jump_flag,
            cols.opcode_jump_if_zero_flag,
            cols.opcode_jump_if_flag,
            cols.opcode_skip_flag,
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            flags.iter().zip(JumpOpcode::iter()).fold(
                AB::Expr::ZERO,
                |acc, (flag, local_opcode)| {
                    acc + (*flag).into() * AB::Expr::from_canonical_u8(local_opcode as u8)
                },
            ),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [[AB::Expr::ZERO; RV32_REGISTER_NUM_LIMBS]].into(),
            writes: [].into(), // No writes for jump instructions
            instruction: JumpInstruction {
                is_valid,
                opcode: expected_opcode,
                is_jump: cols.opcode_jump_flag.into(),
                is_jump_if_zero: cols.opcode_jump_if_zero_flag.into(),
                is_jump_if: cols.opcode_jump_if_flag.into(),
                is_skip: cols.opcode_skip_flag.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        JumpOpcode::CLASS_OFFSET
    }
}

#[derive(Default)]
pub struct JumpCoreChipWom {
    pub air: JumpCoreAir,
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChipWom<F, I> for JumpCoreChipWom
where
    I::Reads: Into<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 0]>,
{
    type Record = ();
    type Air = JumpCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        _from_fp: u32,
        _reads: I::Reads,
    ) -> eyre::Result<(AdapterRuntimeContextWom<F, I>, Self::Record)> {
        let output = AdapterRuntimeContextWom {
            to_pc: None,
            to_fp: None,
            writes: [].into(),
        };

        Ok((output, ()))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            JumpOpcode::from_usize(opcode - JumpOpcode::CLASS_OFFSET)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], _record: Self::Record) {
        let _core_cols: &mut JumpCoreCols<F> = row_slice.borrow_mut();
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

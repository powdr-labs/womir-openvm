use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::{
    AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
    VmCoreAir, VmCoreChip,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use openvm_womir_transpiler::JaafOpcode::{self, *};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::adapters::RV32_REGISTER_NUM_LIMBS;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct JaafCoreCols<T> {
    pub is_valid: T,
}

#[derive(Default, Debug, Clone)]
pub struct JaafCoreAir {}

impl<F: Field> BaseAir<F> for JaafCoreAir {
    fn width(&self) -> usize {
        JaafCoreCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for JaafCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        JaafCoreCols::<F>::struct_reflection()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for JaafCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for JaafCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 2]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &JaafCoreCols<AB::Var> = (*local_core).borrow();
        let JaafCoreCols::<AB::Var> { is_valid } = *cols;

        builder.assert_bool(is_valid);

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, JAAF);

        AdapterAirContext {
            to_pc: None,
            reads: [
                [AB::Expr::ZERO; RV32_REGISTER_NUM_LIMBS],
                [AB::Expr::ZERO; RV32_REGISTER_NUM_LIMBS],
            ]
            .into(),
            writes: [
                [AB::Expr::ZERO; RV32_REGISTER_NUM_LIMBS],
                [AB::Expr::ZERO; RV32_REGISTER_NUM_LIMBS],
            ]
            .into(),
            instruction: MinimalInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        JaafOpcode::CLASS_OFFSET
    }
}

#[derive(Default)]
pub struct JaafCoreChipWom {
    pub air: JaafCoreAir,
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for JaafCoreChipWom
where
    I::Reads: Into<[[F; RV32_REGISTER_NUM_LIMBS]; 2]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 2]>,
{
    type Record = ();
    type Air = JaafCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        _reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let output = AdapterRuntimeContext {
            to_pc: None,
            writes: [
                [F::ZERO; RV32_REGISTER_NUM_LIMBS],
                [F::ZERO; RV32_REGISTER_NUM_LIMBS],
            ]
            .into(),
        };

        Ok((output, ()))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            JaafOpcode::from_usize(opcode - JaafOpcode::CLASS_OFFSET)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], _record: Self::Record) {
        let core_cols: &mut JaafCoreCols<F> = row_slice.borrow_mut();
        core_cols.is_valid = F::ONE;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

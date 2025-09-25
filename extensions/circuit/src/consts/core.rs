use std::borrow::Borrow;

use openvm_circuit::arch::{
    AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
    VmCoreAir, VmCoreChip,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{LocalOpcode, instruction::Instruction};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use openvm_womir_transpiler::ConstOpcodes;
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::adapters::RV32_REGISTER_NUM_LIMBS;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct ConstsCoreCols<T> {
    pub is_valid: T,
}

#[derive(Debug, Clone)]
pub struct ConstsCoreAir {}

impl<F: Field> BaseAir<F> for ConstsCoreAir {
    fn width(&self) -> usize {
        ConstsCoreCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for ConstsCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        ConstsCoreCols::<F>::struct_reflection()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for ConstsCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for ConstsCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 0]; 0]>,
    I::Writes: From<[[AB::Expr; 0]; 0]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let core_cols: &ConstsCoreCols<_> = local_core.borrow();

        // Need at least one constraint otherwise stark-backend complains.
        builder.assert_bool(core_cols.is_valid);

        let opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            AB::Expr::from_canonical_usize(ConstOpcodes::CONST32 as usize),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [].into(),
            writes: [].into(),
            instruction: MinimalInstruction {
                is_valid: core_cols.is_valid.into(),
                opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        ConstOpcodes::CLASS_OFFSET
    }
}

pub struct ConstsCoreChipWom {
    pub air: ConstsCoreAir,
}

impl ConstsCoreChipWom {
    pub fn new() -> Self {
        Self {
            air: ConstsCoreAir {},
        }
    }
}

impl Default for ConstsCoreChipWom {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for ConstsCoreChipWom
where
    I::Reads: Into<[[F; RV32_REGISTER_NUM_LIMBS]; 0]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
{
    type Record = ();
    type Air = ConstsCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        _reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let output = AdapterRuntimeContext {
            to_pc: None,
            writes: [[F::ZERO; 4]].into(),
        };

        Ok((output, ()))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("CONSTS_{}", opcode - ConstOpcodes::CLASS_OFFSET)
    }

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {}

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

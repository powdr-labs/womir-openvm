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
use openvm_womir_transpiler::RegWriteOpcode;
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::adapters::RV32_REGISTER_NUM_LIMBS;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct RegWriteCoreCols<T> {
    pub is_valid: T,
}

#[derive(Debug, Clone)]
pub struct RegWriteCoreAir {}

impl<F: Field> BaseAir<F> for RegWriteCoreAir {
    fn width(&self) -> usize {
        RegWriteCoreCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for RegWriteCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        RegWriteCoreCols::<F>::struct_reflection()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for RegWriteCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for RegWriteCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 0]; 0]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let core_cols: &RegWriteCoreCols<_> = local_core.borrow();

        // Need at least one constraint otherwise stark-backend complains.
        builder.assert_bool(core_cols.is_valid);

        let opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            AB::Expr::from_canonical_usize(RegWriteOpcode::CONST32 as usize),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [].into(),
            writes: [[AB::Expr::ZERO; RV32_REGISTER_NUM_LIMBS]].into(),
            instruction: MinimalInstruction {
                is_valid: core_cols.is_valid.into(),
                opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        RegWriteOpcode::CLASS_OFFSET
    }
}

pub struct RegWriteCoreChipWom {
    pub air: RegWriteCoreAir,
}

impl RegWriteCoreChipWom {
    pub fn new() -> Self {
        Self {
            air: RegWriteCoreAir {},
        }
    }
}

impl Default for RegWriteCoreChipWom {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for RegWriteCoreChipWom
where
    I::Reads: Into<[[F; RV32_REGISTER_NUM_LIMBS]; 0]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
{
    type Record = ();
    type Air = RegWriteCoreAir;

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
        format!("REG_WRITE_{}", opcode - RegWriteOpcode::CLASS_OFFSET)
    }

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {}

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

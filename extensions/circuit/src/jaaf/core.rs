use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::{
    AdapterAirContext, AdapterRuntimeContext, Result, VmAdapterInterface, VmCoreAir, VmCoreChip,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{LocalOpcode, instruction::Instruction};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use openvm_womir_transpiler::JaafOpcode;
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::adapters::{JaafInstruction, RV32_REGISTER_NUM_LIMBS};

use strum::IntoEnumIterator;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct JaafCoreCols<T> {
    pub opcode_jaaf_flag: T,
    pub opcode_jaaf_save_flag: T,
    pub opcode_ret_flag: T,
    pub opcode_call_flag: T,
    pub opcode_call_indirect_flag: T,
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
    I::ProcessedInstruction: From<JaafInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &JaafCoreCols<AB::Var> = (*local_core).borrow();
        let flags = [
            cols.opcode_jaaf_flag,
            cols.opcode_jaaf_save_flag,
            cols.opcode_ret_flag,
            cols.opcode_call_flag,
            cols.opcode_call_indirect_flag,
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            flags.iter().zip(JaafOpcode::iter()).fold(
                AB::Expr::ZERO,
                |acc, (flag, local_opcode)| {
                    acc + (*flag).into() * AB::Expr::from_canonical_u8(local_opcode as u8)
                },
            ),
        );

        let save_pc = cols.opcode_call_flag + cols.opcode_call_indirect_flag;
        let save_fp =
            cols.opcode_call_flag + cols.opcode_call_indirect_flag + cols.opcode_jaaf_save_flag;
        let read_pc = cols.opcode_ret_flag + cols.opcode_call_indirect_flag;

        AdapterAirContext {
            to_pc: None,
            reads: [
                [AB::Expr::ZERO; RV32_REGISTER_NUM_LIMBS], // pc
                [AB::Expr::ZERO; RV32_REGISTER_NUM_LIMBS], // fp
            ]
            .into(),
            writes: [
                [AB::Expr::ZERO; RV32_REGISTER_NUM_LIMBS], // pc
                [AB::Expr::ZERO; RV32_REGISTER_NUM_LIMBS], // fp
            ]
            .into(),
            instruction: JaafInstruction {
                is_valid,
                opcode: expected_opcode,
                save_pc,
                save_fp,
                read_pc,
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
        let _core_cols: &mut JaafCoreCols<F> = row_slice.borrow_mut();
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

use std::borrow::Borrow;

use openvm_circuit::arch::{AdapterAirContext, VmCoreAir};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use openvm_womir_transpiler::JaafOpcode;
use struct_reflection::{StructReflection, StructReflectionHelper};
use strum::IntoEnumIterator;

use crate::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::adapters::jaaf::{JaafAdapterInterface, JaafInstruction};

/// Core columns for the JAAF chip.
///
/// The core is responsible for:
/// - Opcode flag decoding (exactly one flag set per valid row)
/// - Decomposing old FP and return PC into bytes for save writes
/// - Composing new FP from read data for the FP_AS write
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct JaafCoreCols<T> {
    /// Data read from to_fp_reg (new FP as 4 bytes)
    pub new_fp_data: [T; RV32_REGISTER_NUM_LIMBS],
    /// Data read from to_pc_reg (target PC as 4 bytes, valid only for RET/CALL_INDIRECT)
    pub to_pc_data: [T; RV32_REGISTER_NUM_LIMBS],

    /// Old FP as 4 bytes (for save_fp writes). Decomposition of from_state.fp.
    pub old_fp_data: [T; RV32_REGISTER_NUM_LIMBS],
    /// Return PC as 4 bytes (for save_pc writes). Decomposition of from_pc + DEFAULT_PC_STEP.
    pub return_pc_data: [T; RV32_REGISTER_NUM_LIMBS],

    /// Opcode flags (exactly one is 1 per valid row)
    pub is_jaaf: T,
    pub is_jaaf_save: T,
    pub is_ret: T,
    pub is_call: T,
    pub is_call_indirect: T,
}

/// Core record written by the preflight executor during trace generation.
/// This is overlaid on the same memory as JaafCoreCols during trace filling.
///
/// IMPORTANT: Fields must be in the same order as JaafCoreCols for the
/// reverse-order filling trick to work correctly (AlignedBytesBorrow layout).
#[repr(C)]
#[derive(openvm_circuit_primitives::AlignedBytesBorrow, Debug, Clone, Copy)]
pub struct JaafCoreRecord {
    pub new_fp_data: [u8; RV32_REGISTER_NUM_LIMBS],
    pub to_pc_data: [u8; RV32_REGISTER_NUM_LIMBS],
    pub old_fp_data: [u8; RV32_REGISTER_NUM_LIMBS],
    pub return_pc_data: [u8; RV32_REGISTER_NUM_LIMBS],
    pub local_opcode: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct JaafCoreAir {
    pub offset: usize,
}

impl JaafCoreAir {
    pub fn new(offset: usize) -> Self {
        Self { offset }
    }
}

impl<F: Field> BaseAir<F> for JaafCoreAir {
    fn width(&self) -> usize {
        JaafCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for JaafCoreAir {}

impl<F: Field> ColumnsAir<F> for JaafCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        JaafCoreCols::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmCoreAir<AB, JaafAdapterInterface<AB>> for JaafCoreAir {
    fn start_offset(&self) -> usize {
        self.offset
    }

    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, JaafAdapterInterface<AB>> {
        let cols: &JaafCoreCols<AB::Var> = local_core.borrow();

        let flags = [
            cols.is_jaaf,
            cols.is_jaaf_save,
            cols.is_ret,
            cols.is_call,
            cols.is_call_indirect,
        ];

        // Each flag is boolean, and at most one is 1
        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        // Constrain return_pc_data = decompose(from_pc + DEFAULT_PC_STEP)
        let return_pc_composed = cols
            .return_pc_data
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &limb)| {
                acc + limb.into() * AB::Expr::from_canonical_u32(1u32 << (i * 8))
            });
        builder.when(is_valid.clone()).assert_eq(
            return_pc_composed,
            from_pc + AB::Expr::from_canonical_u32(openvm_instructions::program::DEFAULT_PC_STEP),
        );

        // Compose new_fp for the FP_AS write
        let new_fp_composed = cols
            .new_fp_data
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &limb)| {
                acc + limb.into() * AB::Expr::from_canonical_u32(1u32 << (i * 8))
            });

        // old_fp_data decomposition is constrained by the adapter:
        // adapter checks compose(old_fp_data) == from_state.fp

        // Build the opcode expression from flags
        let expected_opcode = flags.iter().zip(JaafOpcode::iter()).fold(
            AB::Expr::ZERO,
            |acc, (&flag, local_opcode)| {
                acc + flag.into() * AB::Expr::from_canonical_usize(local_opcode as usize)
            },
        );
        let expected_opcode = expected_opcode + AB::Expr::from_canonical_usize(self.offset);

        // Derive conditional flags from opcode indicator variables
        let has_pc_read: AB::Expr = cols.is_ret.into() + cols.is_call_indirect.into();
        let has_save_fp: AB::Expr =
            cols.is_jaaf_save.into() + cols.is_call.into() + cols.is_call_indirect.into();
        let has_save_pc: AB::Expr = cols.is_call.into() + cols.is_call_indirect.into();

        AdapterAirContext {
            to_pc: None, // PC is handled by the adapter
            reads: [
                cols.new_fp_data.map(Into::into),
                cols.to_pc_data.map(Into::into),
            ],
            writes: (
                [
                    cols.old_fp_data.map(Into::into),
                    cols.return_pc_data.map(Into::into),
                ],
                [new_fp_composed],
            ),
            instruction: JaafInstruction {
                is_valid: is_valid.clone(),
                opcode: expected_opcode,
                has_pc_read,
                has_save_fp,
                has_save_pc,
            },
        }
    }
}

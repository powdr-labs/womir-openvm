use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::{
    AdapterAirContext, MinimalInstruction, Result, VmAdapterInterface, VmCoreAir,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use openvm_womir_transpiler::AllocateFrameOpcode;
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{AdapterRuntimeContextWom, VmCoreChipWom};

use crate::adapters::RV32_REGISTER_NUM_LIMBS;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct AllocateFrameCoreCols<T> {
    pub is_valid: T,
}

#[derive(Default, Debug, Clone)]
pub struct AllocateFrameCoreAir {}

impl<F: Field> BaseAir<F> for AllocateFrameCoreAir {
    fn width(&self) -> usize {
        AllocateFrameCoreCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for AllocateFrameCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        AllocateFrameCoreCols::<F>::struct_reflection()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for AllocateFrameCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for AllocateFrameCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 0]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let core_cols: &AllocateFrameCoreCols<_> = local_core.borrow();

        // Need at least one constraint otherwise stark-backend complains.
        builder.assert_bool(core_cols.is_valid);

        let opcode =
            VmCoreAir::<AB, I>::opcode_to_global_expr(self, AllocateFrameOpcode::ALLOCATE_FRAME);

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
        AllocateFrameOpcode::CLASS_OFFSET
    }
}

#[derive(Default)]
pub struct AllocateFrameCoreChipWom {
    pub air: AllocateFrameCoreAir,
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChipWom<F, I> for AllocateFrameCoreChipWom
where
    I::Reads: Into<[[F; RV32_REGISTER_NUM_LIMBS]; 2]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
{
    type Record = ();
    type Air = AllocateFrameCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        _instruction: &Instruction<F>,
        _from_pc: u32,
        _from_fp: u32,
        _reads: I::Reads,
    ) -> Result<(AdapterRuntimeContextWom<F, I>, Self::Record)> {
        let output = AdapterRuntimeContextWom {
            to_pc: None,
            to_fp: None,
            writes: [[F::ZERO; RV32_REGISTER_NUM_LIMBS]].into(),
        };

        Ok((output, ()))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "ALLOCATE_FRAME_{}",
            opcode - AllocateFrameOpcode::CLASS_OFFSET
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], _record: Self::Record) {
        let core_cols: &mut AllocateFrameCoreCols<F> = row_slice.borrow_mut();
        core_cols.is_valid = F::ONE;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

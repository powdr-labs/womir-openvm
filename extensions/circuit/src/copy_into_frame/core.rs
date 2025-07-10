use std::borrow::Borrow;

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
use openvm_womir_transpiler::CopyIntoFrameOpcode;
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{AdapterRuntimeContextWom, VmCoreChipWom};

use crate::adapters::RV32_REGISTER_NUM_LIMBS;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct CopyIntoFrameCoreCols<T> {
    pub is_valid: T,
}

#[derive(Debug, Clone)]
pub struct CopyIntoFrameCoreAir {}

impl<F: Field> BaseAir<F> for CopyIntoFrameCoreAir {
    fn width(&self) -> usize {
        CopyIntoFrameCoreCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for CopyIntoFrameCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        CopyIntoFrameCoreCols::<F>::struct_reflection()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for CopyIntoFrameCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for CopyIntoFrameCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 0]; 0]>,
    I::Writes: From<[[AB::Expr; 0]; 0]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        _builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let core_cols: &CopyIntoFrameCoreCols<_> = local_core.borrow();

        let zeroes = [AB::F::ZERO; 4];
        let data: [AB::Expr; RV32_REGISTER_NUM_LIMBS] = zeroes.map(|x| x.into());
        let opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            AB::Expr::from_canonical_usize(CopyIntoFrameOpcode::COPY_INTO_FRAME as usize),
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
        CopyIntoFrameOpcode::CLASS_OFFSET
    }
}

pub struct CopyIntoFrameCoreChipWom {
    pub air: CopyIntoFrameCoreAir,
}

impl CopyIntoFrameCoreChipWom {
    pub fn new() -> Self {
        Self {
            air: CopyIntoFrameCoreAir {},
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChipWom<F, I> for CopyIntoFrameCoreChipWom
where
    I::Reads: Into<[[F; RV32_REGISTER_NUM_LIMBS]; 2]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
{
    type Record = ();
    type Air = CopyIntoFrameCoreAir;

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
            writes: [[F::ZERO; 4]].into(),
        };

        Ok((output, ()))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "COPY_INTO_FRAME_{}",
            opcode - CopyIntoFrameOpcode::CLASS_OFFSET
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {}

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

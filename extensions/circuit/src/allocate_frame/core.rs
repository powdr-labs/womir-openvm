use std::{
    borrow::{Borrow, BorrowMut},
    cell::Cell,
};

use openvm_circuit::arch::{
    AdapterAirContext, MinimalInstruction, Result, VmAdapterInterface, VmCoreAir,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use openvm_womir_transpiler::AllocateFrameOpcode;
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{AdapterRuntimeContextWom, VmCoreChipWom};

use crate::adapters::{decompose, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct AllocateFrameCoreCols<T> {
    pub target_reg: T,
    pub amount_imm: T,
    pub allocated_ptr: [T; RV32_REGISTER_NUM_LIMBS],
    pub is_valid: T,
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct AllocateFrameCoreRecord<F> {
    pub target_reg: F,
    pub amount_imm: F,
    pub allocated_ptr: [F; RV32_REGISTER_NUM_LIMBS],
}

#[derive(Debug, Clone)]
pub struct AllocateFrameCoreAir {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,
}

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
        _builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let core_cols: &AllocateFrameCoreCols<_> = local_core.borrow();
        let allocated_ptr: [AB::Expr; RV32_REGISTER_NUM_LIMBS] =
            core_cols.allocated_ptr.map(|x| x.into());

        let opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            AB::Expr::from_canonical_usize(AllocateFrameOpcode::ALLOCATE_FRAME as usize),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [].into(),
            writes: [allocated_ptr].into(),
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

pub struct AllocateFrameCoreChipWom {
    pub air: AllocateFrameCoreAir,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl AllocateFrameCoreChipWom {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        assert!(range_checker_chip.range_max_bits() >= 16);
        Self {
            air: AllocateFrameCoreAir {
                bitwise_lookup_bus: bitwise_lookup_chip.bus(),
                range_bus: range_checker_chip.bus(),
            },
            bitwise_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChipWom<F, I> for AllocateFrameCoreChipWom
where
    I::Reads: Into<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
{
    type Record = AllocateFrameCoreRecord<F>;
    type Air = AllocateFrameCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        _from_fp: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContextWom<F, I>, Self::Record)> {
        let Instruction { a, b, .. } = *instruction;

        // ALLOCATE_FRAME: target_reg (a), amount_imm (b)
        let target_reg = a;

        let allocated_data = reads.into()[0];

        let output = AdapterRuntimeContextWom {
            to_pc: None,
            to_fp: None,
            writes: [allocated_data].into(),
        };

        Ok((
            output,
            AllocateFrameCoreRecord {
                target_reg,
                amount_imm: b,
                allocated_ptr: allocated_data,
            },
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "ALLOCATE_FRAME_{}",
            opcode - AllocateFrameOpcode::CLASS_OFFSET
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let core_cols: &mut AllocateFrameCoreCols<F> = row_slice.borrow_mut();
        core_cols.target_reg = record.target_reg;
        core_cols.amount_imm = record.amount_imm;
        core_cols.allocated_ptr = record.allocated_ptr;
        core_cols.is_valid = F::ONE;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

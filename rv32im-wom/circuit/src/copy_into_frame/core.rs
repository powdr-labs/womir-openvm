use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::{
    AdapterAirContext, MinimalInstruction, Result, VmAdapterInterface, VmCoreAir,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32im_wom_transpiler::Rv32CopyIntoFrameOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{AdapterRuntimeContextWom, VmCoreChipWom};

use crate::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv32CopyIntoFrameCoreCols<T> {
    pub rd: T,
    pub rs1_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub rs2_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub is_valid: T,
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct Rv32CopyIntoFrameCoreRecord<F> {
    pub rd: F,
    pub rs1_data: [F; RV32_REGISTER_NUM_LIMBS],
    pub rs2_data: [F; RV32_REGISTER_NUM_LIMBS],
}

#[derive(Debug, Clone)]
pub struct Rv32CopyIntoFrameCoreAir {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv32CopyIntoFrameCoreAir {
    fn width(&self) -> usize {
        Rv32CopyIntoFrameCoreCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for Rv32CopyIntoFrameCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        Rv32CopyIntoFrameCoreCols::<F>::struct_reflection()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32CopyIntoFrameCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv32CopyIntoFrameCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        _builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let core_cols: &Rv32CopyIntoFrameCoreCols<_> = local_core.borrow();
        let rs1_data: [AB::Expr; RV32_REGISTER_NUM_LIMBS] = core_cols.rs1_data.map(|x| x.into());
        let rs2_data: [AB::Expr; RV32_REGISTER_NUM_LIMBS] = core_cols.rs2_data.map(|x| x.into());

        let opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            AB::Expr::from_canonical_usize(Rv32CopyIntoFrameOpcode::COPY_INTO_FRAME as usize),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [rs1_data.clone(), rs2_data].into(),
            writes: [rs1_data].into(), // Write rs1 data to memory
            instruction: MinimalInstruction {
                is_valid: core_cols.is_valid.into(),
                opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv32CopyIntoFrameOpcode::CLASS_OFFSET
    }
}

pub struct Rv32CopyIntoFrameCoreChipWom {
    pub air: Rv32CopyIntoFrameCoreAir,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl Rv32CopyIntoFrameCoreChipWom {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        assert!(range_checker_chip.range_max_bits() >= 16);
        Self {
            air: Rv32CopyIntoFrameCoreAir {
                bitwise_lookup_bus: bitwise_lookup_chip.bus(),
                range_bus: range_checker_chip.bus(),
            },
            bitwise_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChipWom<F, I> for Rv32CopyIntoFrameCoreChipWom
where
    I::Reads: Into<[[F; RV32_REGISTER_NUM_LIMBS]; 2]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 1]>,
{
    type Record = Rv32CopyIntoFrameCoreRecord<F>;
    type Air = Rv32CopyIntoFrameCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        from_fp: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContextWom<F, I>, Self::Record)> {
        let Instruction { a, .. } = *instruction;

        // COPY_INTO_FRAME: rd (a), rs1 (reads[0]), rs2 (reads[1])
        let reads_array: [[F; RV32_REGISTER_NUM_LIMBS]; 2] = reads.into();
        let rs1_data = reads_array[0]; // Value to copy
        let rs2_data = reads_array[1]; // Frame pointer

        let output = AdapterRuntimeContextWom {
            to_pc: None,
            to_fp: None,
            // writes: [rs1_data, rs2_data].into(),
            writes: [rs1_data].into(),
        };

        Ok((
            output,
            Rv32CopyIntoFrameCoreRecord {
                rd: a,
                rs1_data,
                rs2_data,
            },
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "COPY_INTO_FRAME_{}",
            opcode - Rv32CopyIntoFrameOpcode::CLASS_OFFSET
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let core_cols: &mut Rv32CopyIntoFrameCoreCols<F> = row_slice.borrow_mut();
        core_cols.rd = record.rd;
        core_cols.rs1_data = record.rs1_data;
        core_cols.rs2_data = record.rs2_data;
        core_cols.is_valid = F::ONE;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

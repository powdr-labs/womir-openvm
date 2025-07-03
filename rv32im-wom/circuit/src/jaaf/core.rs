use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::arch::{
    AdapterAirContext, MinimalInstruction, Result, VmAdapterInterface, VmCoreAir,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode,
};
use openvm_rv32im_wom_transpiler::Rv32JaafOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use serde::{Deserialize, Serialize};
use struct_reflection::{StructReflection, StructReflectionHelper};

use crate::{AdapterRuntimeContextWom, VmCoreChipWom};

use crate::adapters::{compose, decompose, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

const RV32_LIMB_MAX: u32 = (1 << RV32_CELL_BITS) - 1;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv32JaafCoreCols<T> {
    pub imm: T,
    pub rs1_data: [T; RV32_REGISTER_NUM_LIMBS],
    // To save a column, we only store the 3 most significant limbs of `rd_data`
    // the least significant limb can be derived using from_pc and the other limbs
    pub rd_data: [T; RV32_REGISTER_NUM_LIMBS - 1],
    pub is_valid: T,

    pub to_pc_least_sig_bit: T,
    /// These are the limbs of `to_pc * 2`.
    pub to_pc_limbs: [T; 2],
}

#[repr(C)]
#[derive(Serialize, Deserialize)]
pub struct Rv32JaafCoreRecord<F> {
    pub imm: F,
    pub rs1_data: [F; RV32_REGISTER_NUM_LIMBS],
    pub rd_data: [F; RV32_REGISTER_NUM_LIMBS - 1],
    pub to_pc_least_sig_bit: F,
    pub to_pc_limbs: [u32; 2],
}

#[derive(Debug, Clone)]
pub struct Rv32JaafCoreAir {
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv32JaafCoreAir {
    fn width(&self) -> usize {
        Rv32JaafCoreCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for Rv32JaafCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        Rv32JaafCoreCols::<F>::struct_reflection()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32JaafCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv32JaafCoreAir
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
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv32JaafCoreCols<AB::Var> = (*local_core).borrow();
        let Rv32JaafCoreCols::<AB::Var> {
            imm,
            rs1_data: pc_source_data,
            rd_data: saved_pc_data,
            is_valid,
            to_pc_least_sig_bit,
            to_pc_limbs,
        } = *cols;

        builder.assert_bool(is_valid);

        // composed is the composition of 3 most significant limbs of saved_pc_data
        let composed = saved_pc_data
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u32(1 << ((i + 1) * RV32_CELL_BITS))
            });

        let least_sig_limb = from_pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP) - composed;

        // saved_pc_limbs is the final decomposition of `from_pc + DEFAULT_PC_STEP` we need.
        // The range check on `least_sig_limb` also ensures that `saved_pc_limbs` correctly represents
        // `from_pc + DEFAULT_PC_STEP`. Specifically, if `saved_pc_limbs` does not match the
        // expected limb, then `least_sig_limb` becomes the real `least_sig_limb` plus the
        // difference between `composed` and the three most significant limbs of `from_pc +
        // DEFAULT_PC_STEP`. In that case, `least_sig_limb` >= 2^RV32_CELL_BITS.
        let saved_pc_limbs = array::from_fn(|i| {
            if i == 0 {
                least_sig_limb.clone()
            } else {
                saved_pc_data[i - 1].into().clone()
            }
        });

        // Constrain saved_pc_limbs
        // Assumes only from_pc in [0,2^PC_BITS) is allowed by program bus
        self.bitwise_lookup_bus
            .send_range(saved_pc_limbs[0].clone(), saved_pc_limbs[1].clone())
            .eval(builder, is_valid);
        self.range_bus
            .range_check(saved_pc_limbs[2].clone(), RV32_CELL_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(saved_pc_limbs[3].clone(), PC_BITS - RV32_CELL_BITS * 3)
            .eval(builder, is_valid);

        // Constrain to_pc_least_sig_bit + 2 * to_pc_limbs = pc_source + imm as a u32 addition
        // RISC-V spec explicitly sets the least significant bit of `to_pc` to 0
        let pc_source_limbs_01 =
            pc_source_data[0] + pc_source_data[1] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);
        let pc_source_limbs_23 =
            pc_source_data[2] + pc_source_data[3] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);
        let inv = AB::F::from_canonical_u32(1 << 16).inverse();

        builder.assert_bool(to_pc_least_sig_bit);
        let carry =
            (pc_source_limbs_01 + imm - to_pc_limbs[0] * AB::F::TWO - to_pc_least_sig_bit) * inv;
        builder.when(is_valid).assert_bool(carry.clone());

        // No sign extension needed since immediates are always positive
        let carry = (pc_source_limbs_23 + carry - to_pc_limbs[1]) * inv;
        builder.when(is_valid).assert_bool(carry);

        // preventing to_pc overflow
        self.range_bus
            .range_check(to_pc_limbs[1], PC_BITS - 16)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(to_pc_limbs[0], 15)
            .eval(builder, is_valid);
        let to_pc =
            to_pc_limbs[0] * AB::F::TWO + to_pc_limbs[1] * AB::F::from_canonical_u32(1 << 16);

        // For now, we'll handle all opcodes as JAAF in the core
        // The adapter will handle the specific behavior
        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, JAAF);

        // For the Air, we need to provide 2 reads and 2 writes
        // pc_source is for PC source (when needed), fp_source is for FP source
        // saved_pc is for PC save, saved_fp is for FP save
        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [
                pc_source_data.map(|x| x.into()),
                pc_source_data.map(|x| x.into()),
            ]
            .into(), // pc_source for PC, fp_source for FP (pc_source used twice in air for now)
            writes: [
                saved_pc_limbs.clone(),
                [AB::Expr::ZERO; RV32_REGISTER_NUM_LIMBS],
            ]
            .into(), // PC save, FP save handled by adapter
            instruction: MinimalInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        Rv32JaafOpcode::CLASS_OFFSET
    }
}

pub struct Rv32JaafCoreChipWom {
    pub air: Rv32JaafCoreAir,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl Rv32JaafCoreChipWom {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        assert!(range_checker_chip.range_max_bits() >= 16);
        Self {
            air: Rv32JaafCoreAir {
                bitwise_lookup_bus: bitwise_lookup_chip.bus(),
                range_bus: range_checker_chip.bus(),
            },
            bitwise_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChipWom<F, I> for Rv32JaafCoreChipWom
where
    I::Reads: Into<[[F; RV32_REGISTER_NUM_LIMBS]; 2]>,
    I::Writes: From<[[F; RV32_REGISTER_NUM_LIMBS]; 2]>,
{
    type Record = Rv32JaafCoreRecord<F>;
    type Air = Rv32JaafCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        from_fp: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContextWom<F, I>, Self::Record)> {
        let Instruction { opcode, d, .. } = *instruction;
        let local_opcode =
            Rv32JaafOpcode::from_usize(opcode.local_opcode_idx(Rv32JaafOpcode::CLASS_OFFSET));

        // For RET and CALL_INDIRECT, the immediate should be 0 as PC comes from register
        let imm = match local_opcode {
            RET | CALL_INDIRECT => 0,
            _ => d.as_canonical_u32(),
        };

        let reads_array: [[F; RV32_REGISTER_NUM_LIMBS]; 2] = reads.into();
        let pc_source_data = reads_array[0];
        let pc_source_val = compose(pc_source_data);

        let (to_pc, rd_data) = run_jalr(local_opcode, from_pc, imm, pc_source_val);

        // For all JAAF instructions, we also need to handle fp
        let fp_source_data = reads_array[1];
        let to_fp = compose(fp_source_data);

        self.bitwise_lookup_chip
            .request_range(rd_data[0], rd_data[1]);
        self.range_checker_chip
            .add_count(rd_data[2], RV32_CELL_BITS);
        self.range_checker_chip
            .add_count(rd_data[3], PC_BITS - RV32_CELL_BITS * 3);

        let mask = (1 << 15) - 1;
        let to_pc_least_sig_bit = pc_source_val.wrapping_add(imm) & 1;

        let to_pc_limbs = array::from_fn(|i| ((to_pc >> (1 + i * 15)) & mask));

        let rd_data = rd_data.map(F::from_canonical_u32);

        // Prepare writes based on opcode
        let writes = match local_opcode {
            JAAF | RET => {
                // No saves, but we still need to provide write data
                [
                    [F::ZERO; RV32_REGISTER_NUM_LIMBS],
                    [F::ZERO; RV32_REGISTER_NUM_LIMBS],
                ]
            }
            JAAF_SAVE => {
                // Save fp to rd2
                [[F::ZERO; RV32_REGISTER_NUM_LIMBS], decompose::<F>(from_fp)]
            }
            CALL | CALL_INDIRECT => {
                // Save pc to rd1 and fp to rd2
                [rd_data, decompose::<F>(from_fp)]
            }
        };

        let output = AdapterRuntimeContextWom {
            to_pc: Some(to_pc),
            to_fp: Some(to_fp),
            writes: writes.into(),
        };

        Ok((
            output,
            Rv32JaafCoreRecord {
                imm: d,
                rd_data: array::from_fn(|i| rd_data[i + 1]),
                rs1_data: pc_source_data,
                to_pc_least_sig_bit: F::from_canonical_u32(to_pc_least_sig_bit),
                to_pc_limbs,
            },
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32JaafOpcode::from_usize(opcode - Rv32JaafOpcode::CLASS_OFFSET)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        self.range_checker_chip.add_count(record.to_pc_limbs[0], 15);
        self.range_checker_chip.add_count(record.to_pc_limbs[1], 14);

        let core_cols: &mut Rv32JaafCoreCols<F> = row_slice.borrow_mut();
        core_cols.imm = record.imm;
        core_cols.rd_data = record.rd_data;
        core_cols.rs1_data = record.rs1_data;
        core_cols.to_pc_least_sig_bit = record.to_pc_least_sig_bit;
        core_cols.to_pc_limbs = record.to_pc_limbs.map(F::from_canonical_u32);
        core_cols.is_valid = F::ONE;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// returns (to_pc, rd_data)
pub(super) fn run_jalr(
    opcode: Rv32JaafOpcode,
    pc: u32,
    imm: u32,
    pc_source: u32,
) -> (u32, [u32; RV32_REGISTER_NUM_LIMBS]) {
    let to_pc = match opcode {
        JAAF | JAAF_SAVE | CALL => {
            // Use immediate for PC
            imm
        }
        RET | CALL_INDIRECT => {
            // Use pc_source for PC directly (no offset)
            let to_pc = pc_source;
            to_pc - (to_pc & 1)
        }
    };
    assert!(to_pc < (1 << PC_BITS));
    (
        to_pc,
        array::from_fn(|i: usize| ((pc + DEFAULT_PC_STEP) >> (RV32_CELL_BITS * i)) & RV32_LIMB_MAX),
    )
}

use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::{
    AdapterAirContext, ImmInstruction, VmAdapterInterface, VmCoreAir, get_record_from_slice,
};
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::{AlignedBorrow, AlignedBytesBorrow};
use openvm_instructions::{LocalOpcode, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use openvm_womir_transpiler::JumpOpcode;
use struct_reflection::{StructReflection, StructReflectionHelper};
use strum::IntoEnumIterator;

use crate::adapters::RV32_REGISTER_NUM_LIMBS;

/// Trace columns for the JUMP core chip.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct JumpCoreCols<T> {
    /// The condition/offset register value (composed from limbs).
    /// For JUMP this is unused (constrained to zero via reads).
    pub rs_val: [T; RV32_REGISTER_NUM_LIMBS],

    /// Immediate value (the target PC for JUMP/JUMP_IF/JUMP_IF_ZERO, unused for SKIP).
    pub imm: T,

    /// Opcode flags (exactly one must be 1).
    pub opcode_jump_flag: T,
    pub opcode_skip_flag: T,
    pub opcode_jump_if_flag: T,
    pub opcode_jump_if_zero_flag: T,

    /// Whether the condition is zero (1 if all limbs are zero, 0 otherwise).
    /// Used for JUMP_IF and JUMP_IF_ZERO.
    pub cond_is_zero: T,

    /// Whether the branch is taken to the immediate target.
    /// - JUMP: branch_taken = 1 (always taken)
    /// - SKIP: branch_taken = 0 (uses relative offset instead)
    /// - JUMP_IF: branch_taken = NOT cond_is_zero
    /// - JUMP_IF_ZERO: branch_taken = cond_is_zero
    pub branch_taken: T,

    /// Inverse of the first non-zero limb (if condition is non-zero).
    /// Used to prove non-zeroness: if cond_is_zero=0, then there exists i such that
    /// rs_val[i] * nonzero_inv_marker[i] = 1.
    pub nonzero_inv_marker: [T; RV32_REGISTER_NUM_LIMBS],
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct JumpCoreAir {
    pub offset: usize,
}

impl<F: Field> BaseAir<F> for JumpCoreAir {
    fn width(&self) -> usize {
        JumpCoreCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for JumpCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        JumpCoreCols::<F>::struct_reflection()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for JumpCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for JumpCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; RV32_REGISTER_NUM_LIMBS]; 1]>,
    I::Writes: Default,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &JumpCoreCols<_> = local.borrow();
        let flags = [
            cols.opcode_jump_flag,
            cols.opcode_skip_flag,
            cols.opcode_jump_if_flag,
            cols.opcode_jump_if_zero_flag,
        ];

        // Exactly one flag must be set.
        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        // Constrain cond_is_zero is boolean.
        builder.assert_bool(cols.cond_is_zero);

        // Constrain cond_is_zero using BranchEqual-style combined sum (max degree 2):
        //   sum = cond_is_zero + sum_i(rs_val[i] * nonzero_inv_marker[i])
        // When cond_is_zero=1: all limbs forced to zero → sum = 1 + 0 = 1.
        // When cond_is_zero=0: sum = 0 + witness_sum, must equal 1 iff value is non-zero.
        let mut sum: AB::Expr = cols.cond_is_zero.into();
        for i in 0..RV32_REGISTER_NUM_LIMBS {
            // When cond_is_zero=1, all limbs must be zero.
            builder.when(cols.cond_is_zero).assert_zero(cols.rs_val[i]);
            sum += cols.rs_val[i].into() * cols.nonzero_inv_marker[i].into();
        }
        // Combined: when(is_valid) * (sum - 1) = 0, degree 1 + 2 = 3.
        builder.when(is_valid.clone()).assert_one(sum);

        // Compute the expected opcode.
        let expected_opcode = flags
            .iter()
            .zip(JumpOpcode::iter())
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            })
            + AB::Expr::from_canonical_usize(self.offset);

        // Constrain branch_taken is boolean.
        builder.assert_bool(cols.branch_taken);

        // Constrain branch_taken per opcode:
        // JUMP: branch_taken = 1 (always taken)
        builder
            .when(cols.opcode_jump_flag)
            .assert_one(cols.branch_taken);
        // SKIP: branch_taken = 0 (uses relative offset formula)
        builder
            .when(cols.opcode_skip_flag)
            .assert_zero(cols.branch_taken);
        // JUMP_IF: branch_taken = NOT cond_is_zero
        builder
            .when(cols.opcode_jump_if_flag)
            .assert_eq(cols.branch_taken, not::<AB::Expr>(cols.cond_is_zero.into()));
        // JUMP_IF_ZERO: branch_taken = cond_is_zero
        builder
            .when(cols.opcode_jump_if_zero_flag)
            .assert_eq(cols.branch_taken, cols.cond_is_zero);

        // Compose the register value into a u32 for SKIP.
        let rs_composed = cols
            .rs_val
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, limb)| {
                acc + (*limb).into()
                    * AB::Expr::from_canonical_u32(1 << (i * crate::adapters::RV32_CELL_BITS))
            });

        // Compute to_pc (all terms are degree 2):
        //   Base:       branch_taken * imm + (1 - branch_taken) * (from_pc + pc_step)
        //   SKIP corr:  opcode_skip_flag * (rs_composed - 1) * pc_step
        //
        // Non-SKIP: base gives correct result; SKIP correction is zero.
        // SKIP (branch_taken=0): base gives from_pc + pc_step;
        //   correction adds (rs_composed - 1) * pc_step → from_pc + rs_composed * pc_step.
        let pc_step = AB::Expr::from_canonical_u32(DEFAULT_PC_STEP);

        let to_pc = cols.branch_taken * cols.imm
            + not::<AB::Expr>(cols.branch_taken.into()) * (from_pc + pc_step.clone())
            + cols.opcode_skip_flag * (rs_composed - AB::Expr::ONE) * pc_step;

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [cols.rs_val.map(Into::into)].into(),
            writes: Default::default(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: cols.imm.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

/// Record for the JUMP core chip.
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct JumpCoreRecord {
    pub rs_val: [u8; RV32_REGISTER_NUM_LIMBS],
    pub imm: u32,
    pub local_opcode: u8,
}

/// Trace filler for the JUMP core chip.
#[derive(Clone, derive_new::new)]
pub struct JumpCoreFiller {
    pub offset: usize,
}

impl JumpCoreFiller {
    pub fn fill_trace_row<F: PrimeField32>(&self, core_row_slice: &mut [F]) {
        let (mut core_row_raw, _) = core_row_slice.split_at_mut(core_row_slice.len());
        let record: &JumpCoreRecord = unsafe { get_record_from_slice(&mut core_row_raw, ()) };
        let core_row: &mut JumpCoreCols<F> = core_row_raw.borrow_mut();

        let local_opcode = JumpOpcode::from_usize(record.local_opcode as usize);

        // Compute cond_is_zero and nonzero_inv_marker.
        let mut cond_is_zero = true;
        let mut nonzero_inv_marker = [F::ZERO; RV32_REGISTER_NUM_LIMBS];
        for (i, &limb) in record.rs_val.iter().enumerate() {
            if limb != 0 {
                cond_is_zero = false;
                nonzero_inv_marker[i] = F::from_canonical_u8(limb).inverse();
                break;
            }
        }

        // Compute branch_taken.
        let branch_taken = match local_opcode {
            JumpOpcode::JUMP => true,
            JumpOpcode::SKIP => false,
            JumpOpcode::JUMP_IF => !cond_is_zero,
            JumpOpcode::JUMP_IF_ZERO => cond_is_zero,
        };

        // Assign in reverse order (since record overlaps with row).
        core_row.nonzero_inv_marker = nonzero_inv_marker;
        core_row.branch_taken = F::from_bool(branch_taken);
        core_row.cond_is_zero = F::from_bool(cond_is_zero);

        core_row.opcode_jump_if_zero_flag = F::from_bool(local_opcode == JumpOpcode::JUMP_IF_ZERO);
        core_row.opcode_jump_if_flag = F::from_bool(local_opcode == JumpOpcode::JUMP_IF);
        core_row.opcode_skip_flag = F::from_bool(local_opcode == JumpOpcode::SKIP);
        core_row.opcode_jump_flag = F::from_bool(local_opcode == JumpOpcode::JUMP);

        core_row.imm = F::from_canonical_u32(record.imm);
        core_row.rs_val = record.rs_val.map(F::from_canonical_u8);
    }
}

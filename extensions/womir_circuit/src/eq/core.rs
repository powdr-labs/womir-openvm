use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{MemoryAuxColsFactory, online::TracingMemory},
};
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::{AlignedBorrow, AlignedBytesBorrow};
use openvm_instructions::{LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use openvm_womir_transpiler::EqOpcode;
use struct_reflection::{StructReflection, StructReflectionHelper};
use strum::IntoEnumIterator;

/// Maximum number of limbs per group for the equality check.
///
/// Each group composes up to GROUP_SIZE limbs with weights 256^i, giving a
/// composed value in [0, 256^GROUP_SIZE - 1]. The difference of two such values
/// is in [-(256^GROUP_SIZE - 1), 256^GROUP_SIZE - 1].
///
/// For soundness, the absolute difference must be less than P/2 (BabyBear prime
/// P ≈ 2^31), so that a non-zero integer difference cannot wrap to zero mod P.
///   GROUP_SIZE=3 → max |diff| = 256^3 - 1 = 16777215 << P/2 ≈ 10^9. ✓
///   GROUP_SIZE=4 → max |diff| = 256^4 - 1 ≈ 4.3×10^9 > P/2. ✗
pub const EQ_GROUP_SIZE: usize = 3;

// ======================== Core Columns ========================

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct EqCoreCols<T, const NUM_LIMBS: usize, const NUM_GROUPS: usize> {
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    /// Boolean result: 1 if condition holds, 0 otherwise.
    pub cmp_result: T,

    pub opcode_eq_flag: T,
    pub opcode_neq_flag: T,

    /// For proving inequality: limbs are grouped into NUM_GROUPS chunks of up to
    /// EQ_GROUP_SIZE. For each group j, the composed difference is:
    ///   group_diff_j = Σ_{i in group_j} (b[i] - c[i]) * 256^(i - group_start)
    /// When b == c, all group differences are zero. When b != c, at least one
    /// group_diff is non-zero. The prover sets the corresponding group_diff_inv
    /// to the inverse of that group_diff (and zero for the others) so that:
    ///   cmp_eq + Σ_j group_diff_j * group_diff_inv_j = 1
    pub group_diff_inv: [T; NUM_GROUPS],
}

// ======================== Core AIR ========================

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct EqCoreAir<const NUM_LIMBS: usize, const NUM_GROUPS: usize> {
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const NUM_GROUPS: usize> BaseAir<F>
    for EqCoreAir<NUM_LIMBS, NUM_GROUPS>
{
    fn width(&self) -> usize {
        EqCoreCols::<F, NUM_LIMBS, NUM_GROUPS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const NUM_GROUPS: usize> ColumnsAir<F>
    for EqCoreAir<NUM_LIMBS, NUM_GROUPS>
{
    fn columns(&self) -> Option<Vec<String>> {
        EqCoreCols::<F, NUM_LIMBS, NUM_GROUPS>::struct_reflection()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const NUM_GROUPS: usize> BaseAirWithPublicValues<F>
    for EqCoreAir<NUM_LIMBS, NUM_GROUPS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const NUM_GROUPS: usize> VmCoreAir<AB, I>
    for EqCoreAir<NUM_LIMBS, NUM_GROUPS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &EqCoreCols<_, NUM_LIMBS, NUM_GROUPS> = local.borrow();
        let flags = [cols.opcode_eq_flag, cols.opcode_neq_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);

        let b = &cols.b;
        let c = &cols.c;

        // 1 if claiming values are equal, 0 otherwise.
        // For EQ: cmp_result=1 means equal → cmp_eq=1.
        // For NEQ: cmp_result=0 means equal → cmp_eq=1.
        let cmp_eq =
            cols.cmp_result * cols.opcode_eq_flag + not(cols.cmp_result) * cols.opcode_neq_flag;

        // Grouped equality proof via multiplicative inverses.
        //
        // Limbs are partitioned into NUM_GROUPS groups of up to EQ_GROUP_SIZE.
        // For each group j, compute the composed difference:
        //   group_diff_j = Σ_{k=0..group_size_j} (b[start+k] - c[start+k]) * 256^k
        //
        // b == c  ⟺  all group_diff_j are zero.
        //
        // Constraints:
        //   cmp_eq + Σ_j group_diff_j * group_diff_inv_j = 1      (when is_valid)
        //   cmp_eq * group_diff_j = 0                               (for all j)
        //
        // When equal: cmp_eq=1, all group_diffs=0, sum=1. ✓
        // When not equal: cmp_eq=0, prover sets one group_diff_inv to the inverse of
        //   the first non-zero group_diff, sum = 0 + 1 = 1. ✓
        let mut sum = cmp_eq.clone();

        for j in 0..NUM_GROUPS {
            let start = j * EQ_GROUP_SIZE;
            let end = std::cmp::min(start + EQ_GROUP_SIZE, NUM_LIMBS);
            let mut group_diff = AB::Expr::ZERO;
            for k in start..end {
                let weight = AB::Expr::from_canonical_u32(1u32 << ((k - start) * 8));
                group_diff += (b[k] - c[k]) * weight;
            }
            sum += group_diff.clone() * cols.group_diff_inv[j];
            builder.assert_zero(cmp_eq.clone() * group_diff);
        }
        builder.when(is_valid.clone()).assert_one(sum);

        let expected_opcode = flags
            .iter()
            .zip(EqOpcode::iter())
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
            })
            + AB::Expr::from_canonical_usize(self.offset);

        // Output: cmp_result in the first limb, zeros elsewhere.
        let mut a: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        a[0] = cols.cmp_result.into();

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [a].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

// ======================== Core Record ========================

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct EqCoreRecord<const NUM_LIMBS: usize> {
    pub b: [u8; NUM_LIMBS],
    pub c: [u8; NUM_LIMBS],
    pub local_opcode: u8,
}

// ======================== Preflight Executor ========================

#[derive(Clone, Copy, derive_new::new)]
pub struct EqExecutorInner<A, const NUM_LIMBS: usize> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, RA, const NUM_LIMBS: usize> PreflightExecutor<F, RA> for EqExecutorInner<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
            'buf,
            EmptyAdapterCoreLayout<F, A>,
            (A::RecordMut<'buf>, &'buf mut EqCoreRecord<NUM_LIMBS>),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", EqOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        core_record.b = rs1;
        core_record.c = rs2;
        core_record.local_opcode = opcode.local_opcode_idx(self.offset) as u8;

        let is_eq = core_record.local_opcode == EqOpcode::EQ as u8;
        let cmp_result = if is_eq { rs1 == rs2 } else { rs1 != rs2 };

        let mut output = [0u8; NUM_LIMBS];
        output[0] = cmp_result as u8;

        self.adapter.write(
            state.memory,
            instruction,
            [output].into(),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

// ======================== Trace Filler ========================

#[derive(Clone, derive_new::new)]
pub struct EqFiller<A, const NUM_LIMBS: usize, const NUM_GROUPS: usize> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, const NUM_LIMBS: usize, const NUM_GROUPS: usize> TraceFiller<F>
    for EqFiller<A, NUM_LIMBS, NUM_GROUPS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // EqCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid EqCoreRecord written by the executor
        // during trace generation
        let record: &EqCoreRecord<NUM_LIMBS> = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut EqCoreCols<F, NUM_LIMBS, NUM_GROUPS> = core_row.borrow_mut();

        let is_eq = record.local_opcode == EqOpcode::EQ as u8;
        let (cmp_result, group_diff_inv) =
            run_eq::<F, NUM_LIMBS, NUM_GROUPS>(is_eq, &record.b, &record.c);

        core_row.group_diff_inv = group_diff_inv;

        core_row.opcode_eq_flag = F::from_bool(is_eq);
        core_row.opcode_neq_flag = F::from_bool(!is_eq);
        core_row.cmp_result = F::from_bool(cmp_result);
        // Write c before b: the record overlaps core_row memory, and writing
        // core_row.b (bytes 0..15) would overwrite record.c (bytes 4..7).
        core_row.c = record.c.map(F::from_canonical_u8);
        core_row.b = record.b.map(F::from_canonical_u8);
    }
}

// ======================== Helper ========================

/// Returns (cmp_result, group_diff_inv).
///
/// When values are equal: all group_diff_inv entries are zero.
/// When values differ: the first group with a non-zero composed difference
/// gets its inverse set; all others are zero.
#[inline(always)]
fn run_eq<F, const NUM_LIMBS: usize, const NUM_GROUPS: usize>(
    is_eq: bool,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> (bool, [F; NUM_GROUPS])
where
    F: PrimeField32,
{
    let mut group_diff_inv = [F::ZERO; NUM_GROUPS];

    for j in 0..NUM_GROUPS {
        let start = j * EQ_GROUP_SIZE;
        let end = std::cmp::min(start + EQ_GROUP_SIZE, NUM_LIMBS);

        // Compose the group difference with weights 256^k.
        let mut group_diff: i64 = 0;
        for k in start..end {
            group_diff += ((x[k] as i64) - (y[k] as i64)) << ((k - start) * 8);
        }

        if group_diff != 0 {
            // Found the first non-zero group. Set its inverse and return.
            let group_diff_field = if group_diff > 0 {
                F::from_canonical_u32(group_diff as u32)
            } else {
                F::ZERO - F::from_canonical_u32((-group_diff) as u32)
            };
            group_diff_inv[j] = group_diff_field.inverse();
            return (!is_eq, group_diff_inv);
        }
    }

    // All groups are zero: values are equal.
    (is_eq, group_diff_inv)
}

use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::{AlignedBorrow, AlignedBytesBorrow};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::{BaseAirWithPublicValues, ColumnsAir},
};
use openvm_womir_transpiler::EqOpcode;
use struct_reflection::{StructReflection, StructReflectionHelper};
use strum::IntoEnumIterator;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct EqCoreCols<T, const NUM_LIMBS: usize> {
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub cmp_result: T,

    pub opcode_eq_flag: T,
    pub opcode_neq_flag: T,

    pub diff_inv_marker: [T; NUM_LIMBS],
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct EqCoreAir<const NUM_LIMBS: usize> {
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize> BaseAir<F> for EqCoreAir<NUM_LIMBS> {
    fn width(&self) -> usize {
        EqCoreCols::<F, NUM_LIMBS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize> ColumnsAir<F> for EqCoreAir<NUM_LIMBS> {
    fn columns(&self) -> Option<Vec<String>> {
        EqCoreCols::<F, NUM_LIMBS>::struct_reflection()
    }
}

impl<F: Field, const NUM_LIMBS: usize> BaseAirWithPublicValues<F> for EqCoreAir<NUM_LIMBS> {}

impl<AB, I, const NUM_LIMBS: usize> VmCoreAir<AB, I> for EqCoreAir<NUM_LIMBS>
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
        let cols: &EqCoreCols<_, NUM_LIMBS> = local.borrow();
        let flags = [cols.opcode_eq_flag, cols.opcode_neq_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);

        let cmp_eq =
            cols.cmp_result * cols.opcode_eq_flag + not(cols.cmp_result) * cols.opcode_neq_flag;
        let mut sum = cmp_eq.clone();

        for i in 0..NUM_LIMBS {
            sum += (cols.b[i] - cols.c[i]) * cols.diff_inv_marker[i];
            builder.assert_zero(cmp_eq.clone() * (cols.b[i] - cols.c[i]));
        }
        builder.when(is_valid.clone()).assert_one(sum);

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            flags
                .iter()
                .zip(EqOpcode::iter())
                .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                    acc + (*flag).into() * AB::Expr::from_canonical_u8(opcode as u8)
                }),
        );

        let mut result = [AB::Expr::ZERO; NUM_LIMBS];
        result[0] = cols.cmp_result.into();

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [result].into(),
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

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct EqCoreRecord<const NUM_LIMBS: usize> {
    pub b: [u8; NUM_LIMBS],
    pub c: [u8; NUM_LIMBS],
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct EqFiller<A, const NUM_LIMBS: usize> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, const NUM_LIMBS: usize> TraceFiller<F> for EqFiller<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &EqCoreRecord<NUM_LIMBS> = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut EqCoreCols<F, NUM_LIMBS> = core_row.borrow_mut();

        let (cmp_result, diff_idx, diff_inv_val) = run_eq::<F, NUM_LIMBS>(
            record.local_opcode == EqOpcode::EQ as u8,
            &record.b,
            &record.c,
        );
        core_row.diff_inv_marker = [F::ZERO; NUM_LIMBS];
        core_row.diff_inv_marker[diff_idx] = diff_inv_val;

        core_row.opcode_neq_flag = F::from_bool(record.local_opcode == EqOpcode::NEQ as u8);
        core_row.opcode_eq_flag = F::from_bool(record.local_opcode == EqOpcode::EQ as u8);

        core_row.cmp_result = F::from_bool(cmp_result);
        // Write c before b: the record bytes overlap with the core columns in memory
        // (record.c occupies the same bytes as core_row.b[1]), so we must write in
        // reverse field order to avoid overwriting record data before reading it.
        core_row.c = record.c.map(F::from_canonical_u8);
        core_row.b = record.b.map(F::from_canonical_u8);
    }
}

#[inline(always)]
pub(super) fn run_eq<F: Field, const NUM_LIMBS: usize>(
    is_eq_opcode: bool,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> (bool, usize, F) {
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        if xi != yi {
            return (
                !is_eq_opcode,
                i,
                (F::from_canonical_u8(xi) - F::from_canonical_u8(yi)).inverse(),
            );
        }
    }
    (is_eq_opcode, 0, F::ZERO)
}

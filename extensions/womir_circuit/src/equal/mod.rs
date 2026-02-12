use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::{
    AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller, EmptyAdapterCoreLayout,
    ExecutionError, MinimalInstruction, PreflightExecutor, RecordArena, TraceFiller,
    VmAdapterInterface, VmAirWrapper, VmChipWrapper, VmCoreAir, VmStateMut, get_record_from_slice,
};
use openvm_circuit::system::memory::{MemoryAuxColsFactory, online::TracingMemory};
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

use super::adapters::{
    BaseAluAdapterAir, BaseAluAdapterExecutor, BaseAluAdapterFiller, RV32_CELL_BITS,
    RV32_REGISTER_NUM_LIMBS, Rv32BaseAluAdapterAir, Rv32BaseAluAdapterExecutor,
    Rv32BaseAluAdapterFiller,
};

mod execution;
pub use execution::EqualExecutor;

// ============ Type Aliases ============

// 32-bit
pub type Rv32EqualAir = VmAirWrapper<Rv32BaseAluAdapterAir, EqualCoreAir<RV32_REGISTER_NUM_LIMBS>>;
pub type Rv32EqualExecutor =
    EqualExecutor<Rv32BaseAluAdapterExecutor<RV32_CELL_BITS>, RV32_REGISTER_NUM_LIMBS>;
pub type Rv32EqualChip<F> = VmChipWrapper<
    F,
    EqualFiller<Rv32BaseAluAdapterFiller<RV32_CELL_BITS>, RV32_REGISTER_NUM_LIMBS>,
>;

// 64-bit
pub type Equal64Air = VmAirWrapper<BaseAluAdapterAir<8, 2>, EqualCoreAir<8>>;
pub type Equal64Executor = EqualExecutor<BaseAluAdapterExecutor<8, 2, RV32_CELL_BITS>, 8>;
pub type Equal64Chip<F> = VmChipWrapper<F, EqualFiller<BaseAluAdapterFiller<2, RV32_CELL_BITS>, 8>>;

// ============ Core Columns ============

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct EqualCoreCols<T, const NUM_LIMBS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub cmp_result: T,
    pub opcode_eq_flag: T,
    pub opcode_neq_flag: T,
    pub diff_inv_marker: [T; NUM_LIMBS],
}

// ============ Core AIR ============

#[derive(Copy, Clone, Debug)]
pub struct EqualCoreAir<const NUM_LIMBS: usize> {
    pub offset: usize,
}

impl<const NUM_LIMBS: usize> EqualCoreAir<NUM_LIMBS> {
    pub fn new(offset: usize) -> Self {
        Self { offset }
    }
}

impl<F: Field, const NUM_LIMBS: usize> BaseAir<F> for EqualCoreAir<NUM_LIMBS> {
    fn width(&self) -> usize {
        EqualCoreCols::<F, NUM_LIMBS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize> ColumnsAir<F> for EqualCoreAir<NUM_LIMBS> {
    fn columns(&self) -> Option<Vec<String>> {
        EqualCoreCols::<F, NUM_LIMBS>::struct_reflection()
    }
}

impl<F: Field, const NUM_LIMBS: usize> BaseAirWithPublicValues<F> for EqualCoreAir<NUM_LIMBS> {}

impl<AB, I, const NUM_LIMBS: usize> VmCoreAir<AB, I> for EqualCoreAir<NUM_LIMBS>
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
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &EqualCoreCols<_, NUM_LIMBS> = local_core.borrow();
        let flags = [cols.opcode_eq_flag, cols.opcode_neq_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);

        let a = &cols.a;
        let b = &cols.b;
        let inv_marker = &cols.diff_inv_marker;

        // 1 if cmp_result indicates a and b are equal, 0 otherwise.
        // For EQ: cmp_result=1 means equal (so cmp_eq=1).
        // For NEQ: cmp_result=1 means not equal (so cmp_eq=0).
        let cmp_eq =
            cols.cmp_result * cols.opcode_eq_flag + not(cols.cmp_result) * cols.opcode_neq_flag;
        let mut sum = cmp_eq.clone();

        // If cmp_eq=1 (claiming equal): all a[i]-b[i] must be 0.
        // If cmp_eq=0 (claiming not equal): exactly one inv_marker[i]*(a[i]-b[i])=1.
        // Combined: sum = cmp_eq + Σ (a[i]-b[i])*inv_marker[i] must equal 1 when is_valid.
        for i in 0..NUM_LIMBS {
            sum += (a[i] - b[i]) * inv_marker[i];
            builder.assert_zero(cmp_eq.clone() * (a[i] - b[i]));
        }
        builder.when(is_valid.clone()).assert_one(sum);

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            flags
                .iter()
                .zip(EqOpcode::iter())
                .fold(AB::Expr::ZERO, |acc, (flag, local_opcode)| {
                    acc + (*flag).into() * AB::Expr::from_canonical_u8(local_opcode as u8)
                }),
        );

        // Write result: [cmp_result, 0, 0, ..., 0]
        let mut result: [AB::Expr; NUM_LIMBS] = std::array::from_fn(|_| AB::Expr::ZERO);
        result[0] = cols.cmp_result.into();

        AdapterAirContext {
            to_pc: None,
            reads: [cols.a.map(Into::into), cols.b.map(Into::into)].into(),
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

// ============ Core Record ============

#[repr(C, align(4))]
#[derive(AlignedBytesBorrow, Debug)]
pub struct EqualCoreRecord<const NUM_LIMBS: usize> {
    pub a: [u8; NUM_LIMBS],
    pub b: [u8; NUM_LIMBS],
    pub local_opcode: u8,
}

// ============ Executor Inner (PreflightExecutor) ============

#[derive(Clone, Copy, derive_new::new)]
pub struct EqualExecutorInner<A, const NUM_LIMBS: usize> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, RA, const NUM_LIMBS: usize> PreflightExecutor<F, RA> for EqualExecutorInner<A, NUM_LIMBS>
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
            (A::RecordMut<'buf>, &'buf mut EqualCoreRecord<NUM_LIMBS>),
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
        let local_opcode = EqOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.a, core_record.b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.local_opcode = local_opcode as u8;

        let is_equal = core_record.a == core_record.b;
        let cmp_result = match local_opcode {
            EqOpcode::EQ => is_equal,
            EqOpcode::NEQ => !is_equal,
        };

        let mut rd = [0u8; NUM_LIMBS];
        rd[0] = cmp_result as u8;

        self.adapter
            .write(state.memory, instruction, [rd].into(), &mut adapter_record);
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

// ============ Filler (TraceFiller) ============

#[derive(derive_new::new)]
pub struct EqualFiller<A, const NUM_LIMBS: usize> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, const NUM_LIMBS: usize> TraceFiller<F> for EqualFiller<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &EqualCoreRecord<NUM_LIMBS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut EqualCoreCols<F, NUM_LIMBS> = core_row.borrow_mut();

        let is_eq = record.local_opcode == EqOpcode::EQ as u8;
        let (cmp_result, diff_idx, diff_inv_val) =
            run_eq::<F, NUM_LIMBS>(is_eq, &record.a, &record.b);

        // Fill in reverse struct order to avoid overlapping writes with the record
        core_row.diff_inv_marker = [F::ZERO; NUM_LIMBS];
        core_row.diff_inv_marker[diff_idx] = diff_inv_val;
        core_row.opcode_neq_flag = F::from_bool(record.local_opcode == EqOpcode::NEQ as u8);
        core_row.opcode_eq_flag = F::from_bool(record.local_opcode == EqOpcode::EQ as u8);
        core_row.cmp_result = F::from_bool(cmp_result);
        core_row.b = record.b.map(F::from_canonical_u8);
        core_row.a = record.a.map(F::from_canonical_u8);
    }
}

// ============ Helper functions ============

/// Compute equality result and the diff_inv_marker data.
/// Returns (cmp_result, diff_idx, diff_inv_val).
#[inline(always)]
fn run_eq<F: PrimeField32, const NUM_LIMBS: usize>(
    is_eq: bool,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> (bool, usize, F) {
    for i in 0..NUM_LIMBS {
        if x[i] != y[i] {
            return (
                !is_eq,
                i,
                (F::from_canonical_u8(x[i]) - F::from_canonical_u8(y[i])).inverse(),
            );
        }
    }
    (is_eq, 0, F::ZERO)
}

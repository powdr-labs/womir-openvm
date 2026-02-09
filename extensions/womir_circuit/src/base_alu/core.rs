use std::{array, borrow::BorrowMut, iter::zip};

use openvm_circuit::{
    arch::*,
    system::memory::{MemoryAuxColsFactory, online::TracingMemory},
};
use openvm_circuit_primitives::{
    AlignedBytesBorrow, bitwise_op_lookup::SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_rv32im_circuit::BaseAluCoreCols;
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[repr(C, align(4))]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BaseAluCoreRecord<const NUM_LIMBS: usize> {
    pub b: [u8; NUM_LIMBS],
    pub c: [u8; NUM_LIMBS],
    // Use u8 instead of usize for better packing
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BaseAluExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(derive_new::new)]
pub struct BaseAluFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
    pub offset: usize,
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for BaseAluExecutor<A, NUM_LIMBS, LIMB_BITS>
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
            (A::RecordMut<'buf>, &'buf mut BaseAluCoreRecord<NUM_LIMBS>),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", BaseAluOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let local_opcode = BaseAluOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.b, core_record.c] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let rd = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &core_record.b, &core_record.c);

        core_record.local_opcode = local_opcode as u8;

        self.adapter
            .write(state.memory, instruction, [rd].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for BaseAluFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // BaseAluCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid BaseAluCoreRecord written by the executor
        // during trace generation
        let record: &BaseAluCoreRecord<NUM_LIMBS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut BaseAluCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();
        // SAFETY: the following is highly unsafe. We are going to cast `core_row` to a record
        // buffer, and then do an _overlapping_ write to the `core_row` as a row of field elements.
        // This requires:
        // - Cols and Record structs should be repr(C) and we write in reverse order (to ensure
        //   non-overlapping)
        // - Do not overwrite any reference in `record` before it has already been used or moved
        // - alignment of `F` must be >= alignment of Record (AlignedBytesBorrow will panic
        //   otherwise)

        let local_opcode = BaseAluOpcode::from_usize(record.local_opcode as usize);
        let a = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &record.b, &record.c);
        // PERF: needless conversion
        core_row.opcode_and_flag = F::from_bool(local_opcode == BaseAluOpcode::AND);
        core_row.opcode_or_flag = F::from_bool(local_opcode == BaseAluOpcode::OR);
        core_row.opcode_xor_flag = F::from_bool(local_opcode == BaseAluOpcode::XOR);
        core_row.opcode_sub_flag = F::from_bool(local_opcode == BaseAluOpcode::SUB);
        core_row.opcode_add_flag = F::from_bool(local_opcode == BaseAluOpcode::ADD);

        if local_opcode == BaseAluOpcode::ADD || local_opcode == BaseAluOpcode::SUB {
            for a_val in a {
                self.bitwise_lookup_chip
                    .request_xor(a_val as u32, a_val as u32);
            }
        } else {
            for (b_val, c_val) in zip(record.b, record.c) {
                self.bitwise_lookup_chip
                    .request_xor(b_val as u32, c_val as u32);
            }
        }
        core_row.c = record.c.map(F::from_canonical_u8);
        core_row.b = record.b.map(F::from_canonical_u8);
        core_row.a = a.map(F::from_canonical_u8);
    }
}

#[inline(always)]
pub(super) fn run_alu<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: BaseAluOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    debug_assert!(LIMB_BITS <= 8, "specialize for bytes");
    match opcode {
        BaseAluOpcode::ADD => run_add::<NUM_LIMBS, LIMB_BITS>(x, y),
        BaseAluOpcode::SUB => run_subtract::<NUM_LIMBS, LIMB_BITS>(x, y),
        BaseAluOpcode::XOR => run_xor::<NUM_LIMBS>(x, y),
        BaseAluOpcode::OR => run_or::<NUM_LIMBS>(x, y),
        BaseAluOpcode::AND => run_and::<NUM_LIMBS>(x, y),
    }
}

#[inline(always)]
fn run_add<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    let mut z = [0u8; NUM_LIMBS];
    let mut carry = [0u8; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let mut overflow =
            (x[i] as u16) + (y[i] as u16) + if i > 0 { carry[i - 1] as u16 } else { 0 };
        carry[i] = (overflow >> LIMB_BITS) as u8;
        overflow &= (1u16 << LIMB_BITS) - 1;
        z[i] = overflow as u8;
    }
    z
}

#[inline(always)]
fn run_subtract<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    let mut z = [0u8; NUM_LIMBS];
    let mut carry = [0u8; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let rhs = y[i] as u16 + if i > 0 { carry[i - 1] as u16 } else { 0 };
        if x[i] as u16 >= rhs {
            z[i] = x[i] - rhs as u8;
            carry[i] = 0;
        } else {
            z[i] = (x[i] as u16 + (1u16 << LIMB_BITS) - rhs) as u8;
            carry[i] = 1;
        }
    }
    z
}

#[inline(always)]
fn run_xor<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS]) -> [u8; NUM_LIMBS] {
    array::from_fn(|i| x[i] ^ y[i])
}

#[inline(always)]
fn run_or<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS]) -> [u8; NUM_LIMBS] {
    array::from_fn(|i| x[i] | y[i])
}

#[inline(always)]
fn run_and<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS]) -> [u8; NUM_LIMBS] {
    array::from_fn(|i| x[i] & y[i])
}

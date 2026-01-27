use openvm_circuit::arch::*;
use openvm_circuit::system::memory::online::TracingMemory;
use openvm_instructions::{LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_rv32im_transpiler::MulOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

// Re-export upstream types that we don't modify
pub use openvm_rv32im_circuit::{
    MultiplicationCoreAir, MultiplicationCoreCols, MultiplicationCoreRecord, MultiplicationFiller,
};

// Our own MultiplicationExecutor that uses FP-aware adapters
#[derive(Clone, Copy, derive_new::new)]
pub struct MultiplicationExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

// FpPreflightExecutor implementation when adapter is FP-aware
impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> crate::FpPreflightExecutor<F, RA>
    for MultiplicationExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + crate::FpAdapterTraceExecutor<
            F,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
            'buf,
            EmptyAdapterCoreLayout<F, A>,
            (
                A::RecordMut<'buf>,
                &'buf mut MultiplicationCoreRecord<NUM_LIMBS, LIMB_BITS>,
            ),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", MulOpcode::from_usize(opcode - self.offset))
    }

    fn execute_with_fp(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
        fp: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        debug_assert_eq!(
            MulOpcode::from_usize(opcode.local_opcode_idx(self.offset)),
            MulOpcode::MUL
        );
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        // Call FP-aware start
        A::start_with_fp(*state.pc, fp, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        // Compute multiplication result
        let (a, _) = run_mul::<NUM_LIMBS, LIMB_BITS>(&rs1, &rs2);

        core_record.b = rs1;
        core_record.c = rs2;

        self.adapter
            .write(state.memory, instruction, [a].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        // FP doesn't change for multiplication operations
        Ok(None)
    }
}

// Helper function to compute multiplication with carry tracking
// Returns (result, carry) where result[i] = (sum_{k=0}^{i} x[k] * y[i-k]) % 2^LIMB_BITS
#[inline(always)]
fn run_mul<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> ([u8; NUM_LIMBS], [u32; NUM_LIMBS]) {
    let mut result = [0u8; NUM_LIMBS];
    let mut carry = [0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let mut res = 0u32;
        if i > 0 {
            res = carry[i - 1];
        }
        for j in 0..=i {
            res += (x[j] as u32) * (y[i - j] as u32);
        }
        carry[i] = res >> LIMB_BITS;
        res %= 1u32 << LIMB_BITS;
        result[i] = res as u8;
    }
    (result, carry)
}

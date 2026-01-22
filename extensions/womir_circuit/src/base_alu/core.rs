use openvm_circuit::arch::*;
use openvm_circuit::system::memory::online::TracingMemory;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

// Re-export upstream types that we don't modify
pub use openvm_rv32im_circuit::{
    BaseAluCoreAir, BaseAluCoreCols, BaseAluCoreRecord, BaseAluFiller,
};

// Our own BaseAluCoreExecutor that uses FP-aware adapters
#[derive(Clone, Copy, derive_new::new)]
pub struct BaseAluCoreExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub adapter: A,
    pub offset: usize,
}

// Helper function for ALU operations (not FP-related, pure ALU logic)
pub fn run_alu<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: BaseAluOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    use BaseAluOpcode::*;
    match opcode {
        ADD => {
            let mut result = [0u8; NUM_LIMBS];
            let mut carry = 0u16;
            for i in 0..NUM_LIMBS {
                let sum = x[i] as u16 + y[i] as u16 + carry;
                result[i] = sum as u8;
                carry = sum >> LIMB_BITS;
            }
            result
        }
        SUB => {
            let mut result = [0u8; NUM_LIMBS];
            let mut borrow = 0i16;
            for i in 0..NUM_LIMBS {
                let diff = x[i] as i16 - y[i] as i16 - borrow;
                result[i] = diff as u8;
                borrow = if diff < 0 { 1 } else { 0 };
            }
            result
        }
        XOR => std::array::from_fn(|i| x[i] ^ y[i]),
        OR => std::array::from_fn(|i| x[i] | y[i]),
        AND => std::array::from_fn(|i| x[i] & y[i]),
    }
}

// InterpreterExecutor - delegates to upstream (required by OpenVM framework, but unused in FP-only system)
impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for BaseAluCoreExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: Clone,
    openvm_rv32im_circuit::BaseAluExecutor<A, NUM_LIMBS, LIMB_BITS>: InterpreterExecutor<F>,
{
    fn pre_compute_size(&self) -> usize {
        openvm_rv32im_circuit::BaseAluExecutor::new(self.adapter.clone(), self.offset).pre_compute_size()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        openvm_rv32im_circuit::BaseAluExecutor::new(self.adapter.clone(), self.offset).pre_compute(pc, inst, data)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        openvm_rv32im_circuit::BaseAluExecutor::new(self.adapter.clone(), self.offset).handler(pc, inst, data)
    }
}

// FpPreflightExecutor implementation when adapter is FP-aware
impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> crate::FpPreflightExecutor<F, RA>
    for BaseAluCoreExecutor<A, NUM_LIMBS, LIMB_BITS>
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
        (A::RecordMut<'buf>, &'buf mut BaseAluCoreRecord<NUM_LIMBS>),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", BaseAluOpcode::from_usize(opcode - self.offset))
    }

    fn execute_with_fp(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
        fp: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        use openvm_instructions::program::DEFAULT_PC_STEP;

        let Instruction { opcode, .. } = instruction;

        let local_opcode = BaseAluOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        // Call FP-aware start
        A::start_with_fp(*state.pc, fp, state.memory, &mut adapter_record);

        [core_record.b, core_record.c] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let rd = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &core_record.b, &core_record.c);

        core_record.local_opcode = local_opcode as u8;

        self.adapter
            .write(state.memory, instruction, [rd].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        // FP doesn't change for basic ALU operations
        Ok(None)
    }
}

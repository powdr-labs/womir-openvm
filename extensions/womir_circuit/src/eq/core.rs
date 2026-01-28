use openvm_circuit::{arch::*, system::memory::online::TracingMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::EqOpcode;

use crate::adapters::RV32_REGISTER_NUM_LIMBS;

// Core executor that implements FpPreflightExecutor
#[derive(Clone, Copy, derive_new::new)]
pub struct EqCoreExecutor<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

// FpPreflightExecutor implementation for Eq
impl<F, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> crate::FpPreflightExecutor<F, RA>
    for EqCoreExecutor<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    RA: RecordArena<'static, (), ()>,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", EqOpcode::from_usize(opcode - self.offset))
    }

    fn execute_with_fp(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
        fp: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        debug_assert!(LIMB_BITS <= 8);
        let Instruction {
            opcode, a, b, c, ..
        } = instruction;

        // Read operands using FP
        let rs1_addr = b.as_canonical_u32() + fp;
        let rs2_addr = c.as_canonical_u32() + fp;

        let (_counter1, rs1) = unsafe {
            state.memory.read::<u8, NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>(
                openvm_instructions::riscv::RV32_REGISTER_AS,
                rs1_addr,
            )
        };
        let (_counter2, rs2) = unsafe {
            state.memory.read::<u8, NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>(
                openvm_instructions::riscv::RV32_REGISTER_AS,
                rs2_addr,
            )
        };

        let local_opcode = opcode.local_opcode_idx(self.offset) as u8;
        let is_eq = local_opcode == EqOpcode::EQ as u8;
        let cmp_result = run_eq::<NUM_LIMBS>(&rs1, &rs2, is_eq);

        let mut output = [0u8; NUM_LIMBS];
        output[0] = cmp_result as u8;

        // Write result using FP
        let rd_addr = a.as_canonical_u32() + fp;
        unsafe {
            state
                .memory
                .write::<u8, NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>(
                    openvm_instructions::riscv::RV32_REGISTER_AS,
                    rd_addr,
                    output,
                );
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        // FP doesn't change for Eq
        Ok(None)
    }
}

// PreCompute struct for InterpreterExecutor
#[repr(C)]
#[derive(AlignedBytesBorrow, Clone, Copy)]
struct EqPreCompute<const NUM_LIMBS: usize> {
    rd: u32,
    rs1: u32,
    rs2: u32,
    is_eq: bool,
}

// Execute function for interpreter mode
unsafe fn execute_eq<
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, openvm_circuit::system::memory::online::GuestMemory, Ctx>,
) {
    use crate::adapters::{memory_read, memory_write};

    let pre_compute = unsafe { &*(pre_compute as *const EqPreCompute<NUM_LIMBS>) };

    // Read operands
    let rs1: [u8; NUM_LIMBS] = memory_read(
        &exec_state.memory,
        openvm_instructions::riscv::RV32_REGISTER_AS,
        pre_compute.rs1,
    );
    let rs2: [u8; NUM_LIMBS] = memory_read(
        &exec_state.memory,
        openvm_instructions::riscv::RV32_REGISTER_AS,
        pre_compute.rs2,
    );

    // Compute comparison
    let cmp_result = run_eq::<NUM_LIMBS>(&rs1, &rs2, pre_compute.is_eq);

    let mut output = [0u8; NUM_LIMBS];
    output[0] = cmp_result as u8;

    // Write result
    memory_write(
        &mut exec_state.memory,
        openvm_instructions::riscv::RV32_REGISTER_AS,
        pre_compute.rd,
        output,
    );

    // Increment PC
    let next_pc = exec_state.pc().wrapping_add(DEFAULT_PC_STEP);
    exec_state.set_pc(next_pc);
}

// InterpreterExecutor implementation
impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for EqCoreExecutor<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<EqPreCompute<NUM_LIMBS>>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let Instruction {
            opcode, a, b, c, ..
        } = *inst;

        let pre_compute = EqPreCompute::<NUM_LIMBS> {
            rd: a.as_canonical_u32(),
            rs1: b.as_canonical_u32(),
            rs2: c.as_canonical_u32(),
            is_eq: opcode.local_opcode_idx(self.offset) == (EqOpcode::EQ as usize),
        };

        data[..std::mem::size_of::<EqPreCompute<NUM_LIMBS>>()].copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                &pre_compute as *const _ as *const u8,
                std::mem::size_of::<EqPreCompute<NUM_LIMBS>>(),
            )
        });

        Ok(execute_eq::<F, Ctx, NUM_LIMBS, LIMB_BITS>)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let Instruction {
            opcode, a, b, c, ..
        } = *inst;

        let pre_compute = EqPreCompute::<NUM_LIMBS> {
            rd: a.as_canonical_u32(),
            rs1: b.as_canonical_u32(),
            rs2: c.as_canonical_u32(),
            is_eq: opcode.local_opcode_idx(self.offset) == (EqOpcode::EQ as usize),
        };

        data[..std::mem::size_of::<EqPreCompute<NUM_LIMBS>>()].copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                &pre_compute as *const _ as *const u8,
                std::mem::size_of::<EqPreCompute<NUM_LIMBS>>(),
            )
        });

        Ok(Box::new(execute_eq::<F, Ctx, NUM_LIMBS, LIMB_BITS>))
    }
}

// Returns true if values are equal (for EQ) or not equal (for NEQ)
#[inline(always)]
fn run_eq<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS], is_eq: bool) -> bool {
    let are_equal = x == y;
    if is_eq { are_equal } else { !are_equal }
}

// Stub types for constraints (not implemented yet)
#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct EqCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub offset: usize,
}

pub type EqCoreCols<T> = std::marker::PhantomData<T>;
pub type EqCoreRecord = ();
pub type EqFiller<Adapter, const NUM_LIMBS: usize, const LIMB_BITS: usize> = Adapter;

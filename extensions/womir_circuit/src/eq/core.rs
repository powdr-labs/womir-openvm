use openvm_circuit::{arch::*, system::memory::online::TracingMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::EqOpcode;
use std::borrow::BorrowMut;
use std::mem::size_of;

use crate::adapters::{RV32_REGISTER_NUM_LIMBS, imm_to_bytes};

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
            opcode, a, b, c, e, ..
        } = instruction;

        // Read first operand using FP
        let rs1_addr = b.as_canonical_u32() + fp;
        let (_counter1, rs1) = unsafe {
            state.memory.read::<u8, NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>(
                openvm_instructions::riscv::RV32_REGISTER_AS,
                rs1_addr,
            )
        };

        // Check if second operand is immediate or register
        let e_u32 = e.as_canonical_u32();
        let rs2 = if e_u32 == RV32_IMM_AS {
            // Immediate value - convert and extend
            let c_u32 = c.as_canonical_u32();
            let imm_bytes = imm_to_bytes(c_u32);
            let mut result = [0u8; NUM_LIMBS];
            result[..4.min(NUM_LIMBS)].copy_from_slice(&imm_bytes[..4.min(NUM_LIMBS)]);
            result
        } else {
            // Read from register using FP
            let rs2_addr = c.as_canonical_u32() + fp;
            let (_counter2, rs2) = unsafe {
                state.memory.read::<u8, NUM_LIMBS, RV32_REGISTER_NUM_LIMBS>(
                    openvm_instructions::riscv::RV32_REGISTER_AS,
                    rs2_addr,
                )
            };
            rs2
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
struct EqPreCompute {
    c: u32,
    a: u8,
    b: u8,
}

// Execute function for interpreter mode
#[inline(always)]
unsafe fn execute_eq<
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
    const IS_IMM: bool,
    const IS_EQ: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, openvm_circuit::system::memory::online::GuestMemory, Ctx>,
) {
    let pre_compute = unsafe { &*(pre_compute as *const EqPreCompute) };

    // Read first operand
    let rs1 = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, pre_compute.b as u32);

    // Read or construct second operand
    let rs2 = if IS_IMM {
        let imm_bytes = imm_to_bytes(pre_compute.c);
        let mut result = [0u8; NUM_LIMBS];
        result[..4.min(NUM_LIMBS)].copy_from_slice(&imm_bytes[..4.min(NUM_LIMBS)]);
        result
    } else {
        exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, pre_compute.c)
    };

    // Compute comparison
    let cmp_result = run_eq::<NUM_LIMBS>(&rs1, &rs2, IS_EQ);

    let mut output = [0u8; NUM_LIMBS];
    output[0] = cmp_result as u8;

    // Write result
    exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &output);

    // Increment PC
    let next_pc = exec_state.pc().wrapping_add(DEFAULT_PC_STEP);
    exec_state.set_pc(next_pc);
}

// Helper for pre-computation
impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> EqCoreExecutor<NUM_LIMBS, LIMB_BITS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut EqPreCompute,
    ) -> Result<(bool, bool), StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let local_opcode = EqOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();

        *data = EqPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok((is_imm, local_opcode == EqOpcode::EQ))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $is_eq:ident) => {
        match ($is_imm, $is_eq) {
            (true, true) => Ok($execute_impl::<_, _, NUM_LIMBS, LIMB_BITS, true, true>),
            (true, false) => Ok($execute_impl::<_, _, NUM_LIMBS, LIMB_BITS, true, false>),
            (false, true) => Ok($execute_impl::<_, _, NUM_LIMBS, LIMB_BITS, false, true>),
            (false, false) => Ok($execute_impl::<_, _, NUM_LIMBS, LIMB_BITS, false, false>),
        }
    };
}

// InterpreterExecutor implementation
impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for EqCoreExecutor<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<EqPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut EqPreCompute = data.borrow_mut();
        let (is_imm, is_eq) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_eq, is_imm, is_eq)
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
        let pre_compute: &mut EqPreCompute = data.borrow_mut();
        let (is_imm, is_eq) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_eq, is_imm, is_eq)
    }
}

// Metered execution function
#[inline(always)]
unsafe fn execute_eq_metered<
    F: PrimeField32,
    Ctx: MeteredExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
    const IS_IMM: bool,
    const IS_EQ: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, openvm_circuit::system::memory::online::GuestMemory, Ctx>,
) {
    use std::borrow::Borrow;
    let pre_compute: &E2PreCompute<EqPreCompute> = unsafe {
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<EqPreCompute>>()).borrow()
    };

    // Track chip height
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);

    let data = &pre_compute.data;

    // Read first operand
    let rs1 = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, data.b as u32);

    // Read or construct second operand
    let rs2 = if IS_IMM {
        let imm_bytes = imm_to_bytes(data.c);
        let mut result = [0u8; NUM_LIMBS];
        result[..4.min(NUM_LIMBS)].copy_from_slice(&imm_bytes[..4.min(NUM_LIMBS)]);
        result
    } else {
        exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, data.c)
    };

    // Compute comparison
    let cmp_result = run_eq::<NUM_LIMBS>(&rs1, &rs2, IS_EQ);

    let mut output = [0u8; NUM_LIMBS];
    output[0] = cmp_result as u8;

    // Write result
    exec_state.vm_write(RV32_REGISTER_AS, data.a as u32, &output);

    // Increment PC
    let next_pc = exec_state.pc().wrapping_add(DEFAULT_PC_STEP);
    exec_state.set_pc(next_pc);
}

macro_rules! dispatch_metered {
    ($execute_impl:ident, $is_imm:ident, $is_eq:ident) => {
        match ($is_imm, $is_eq) {
            (true, true) => Ok($execute_impl::<_, _, NUM_LIMBS, LIMB_BITS, true, true>),
            (true, false) => Ok($execute_impl::<_, _, NUM_LIMBS, LIMB_BITS, true, false>),
            (false, true) => Ok($execute_impl::<_, _, NUM_LIMBS, LIMB_BITS, false, true>),
            (false, false) => Ok($execute_impl::<_, _, NUM_LIMBS, LIMB_BITS, false, false>),
        }
    };
}

// InterpreterMeteredExecutor implementation
impl<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for EqCoreExecutor<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<EqPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<EqPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (is_imm, is_eq) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch_metered!(execute_eq_metered, is_imm, is_eq)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<EqPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (is_imm, is_eq) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch_metered!(execute_eq_metered, is_imm, is_eq)
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

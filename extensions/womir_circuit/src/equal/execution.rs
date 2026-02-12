//! Wrapper executor for the Equal chip.
//! Delegates PreflightExecutor to EqualExecutorInner, and implements
//! InterpreterExecutor/InterpreterMeteredExecutor with FP handling.

use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::memory_config::FpMemory;
use openvm_circuit::{
    arch::*,
    system::memory::online::{GuestMemory, TracingMemory},
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::EqOpcode;

use crate::adapters::{RV32_REGISTER_NUM_LIMBS, imm_to_bytes};

use super::EqualExecutorInner;

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, Copy)]
pub struct EqualExecutor<A, const NUM_LIMBS: usize>(pub EqualExecutorInner<A, NUM_LIMBS>);

impl<A, const NUM_LIMBS: usize> EqualExecutor<A, NUM_LIMBS> {
    pub fn new(adapter: A, offset: usize) -> Self {
        Self(EqualExecutorInner::new(adapter, offset))
    }
}

impl<A, const NUM_LIMBS: usize> std::ops::Deref for EqualExecutor<A, NUM_LIMBS> {
    type Target = EqualExecutorInner<A, NUM_LIMBS>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F, A, RA, const NUM_LIMBS: usize> PreflightExecutor<F, RA> for EqualExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
    EqualExecutorInner<A, NUM_LIMBS>: PreflightExecutor<F, RA>,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        self.0.get_opcode_name(opcode)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        self.0.execute(state, instruction)
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct EqualPreCompute {
    /// Second operand value (if immediate) or register index (if register)
    c: u32,
    /// Result register index
    a: u8,
    /// First operand register index
    b: u8,
    /// Whether second operand is immediate
    is_imm: u8,
}

impl<A, const NUM_LIMBS: usize> EqualExecutor<A, NUM_LIMBS> {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut EqualPreCompute,
    ) -> Result<EqOpcode, StaticProgramError> {
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
        if (d.as_canonical_u32() != RV32_REGISTER_AS)
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        let local_opcode = EqOpcode::from_usize(opcode.local_opcode_idx(self.0.offset));
        *data = EqualPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes::<{ RV32_REGISTER_NUM_LIMBS }>(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            is_imm: is_imm as u8,
        };
        Ok(local_opcode)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $local_opcode:ident, $num_limbs:expr) => {
        Ok(match ($is_imm, $local_opcode) {
            (true, EqOpcode::EQ) => $execute_impl::<_, _, true, { EqOpcode::EQ as u8 }, $num_limbs>,
            (false, EqOpcode::EQ) => {
                $execute_impl::<_, _, false, { EqOpcode::EQ as u8 }, $num_limbs>
            }
            (true, EqOpcode::NEQ) => {
                $execute_impl::<_, _, true, { EqOpcode::NEQ as u8 }, $num_limbs>
            }
            (false, EqOpcode::NEQ) => {
                $execute_impl::<_, _, false, { EqOpcode::NEQ as u8 }, $num_limbs>
            }
        })
    };
}

impl<F, A, const NUM_LIMBS: usize> InterpreterExecutor<F> for EqualExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<EqualPreCompute>()
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
        let data: &mut EqualPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        let is_imm = data.is_imm != 0;

        dispatch!(execute_e1_handler, is_imm, local_opcode, NUM_LIMBS)
    }
}

impl<F, A, const NUM_LIMBS: usize> InterpreterMeteredExecutor<F> for EqualExecutor<A, NUM_LIMBS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<EqualPreCompute>>()
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
        let data: &mut E2PreCompute<EqualPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        let is_imm = data.data.is_imm != 0;

        dispatch!(execute_e2_handler, is_imm, local_opcode, NUM_LIMBS)
    }
}

/// Sign-extend a u32 value to `[u8; N]`.
#[inline(always)]
fn sign_extend_u32<const N: usize>(c: u32) -> [u8; N] {
    let sign_byte = if c & 0x8000_0000 != 0 { 0xFF } else { 0x00 };
    let le = c.to_le_bytes();
    std::array::from_fn(|i| if i < 4 { le[i] } else { sign_byte })
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    const OPCODE: u8,
    const NUM_LIMBS: usize,
>(
    pre_compute: &EqualPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let fp = exec_state.memory.fp::<F>();
    let rs1 = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + (pre_compute.b as u32));
    let rs2 = if IS_IMM {
        sign_extend_u32::<NUM_LIMBS>(pre_compute.c)
    } else {
        exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + pre_compute.c)
    };

    let is_equal = rs1 == rs2;
    let cmp_result = match OPCODE {
        x if x == EqOpcode::EQ as u8 => is_equal,
        x if x == EqOpcode::NEQ as u8 => !is_equal,
        _ => unreachable!(),
    };

    let mut rd = [0u8; NUM_LIMBS];
    rd[0] = cmp_result as u8;

    exec_state.vm_write::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + (pre_compute.a as u32), &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    const OPCODE: u8,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &EqualPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<EqualPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, IS_IMM, OPCODE, NUM_LIMBS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_IMM: bool,
    const OPCODE: u8,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<EqualPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<EqualPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, IS_IMM, OPCODE, NUM_LIMBS>(&pre_compute.data, exec_state);
    }
}

//! This module is largely a copy of extensions/rv32im/circuit/src/less_than/execution.rs.
//! The only differences are:
//! - OpenVM's `LessThanExecutor` is wrapped to allow for trait implementations.
//! - In `execute_e12_impl`, we retrieve and add the frame pointer.
//!
//! This file could be condensed a lot if more of the OpenVM code was public.

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
use openvm_rv32im_circuit::LessThanExecutor as LessThanExecutorInner;
use openvm_rv32im_transpiler::LessThanOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::{RV32_REGISTER_NUM_LIMBS, imm_to_bytes};

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, Copy)]
pub struct LessThanExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    pub LessThanExecutorInner<A, NUM_LIMBS, LIMB_BITS>,
);

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> LessThanExecutor<A, NUM_LIMBS, LIMB_BITS> {
    pub fn new(adapter: A, offset: usize) -> Self {
        Self(LessThanExecutorInner::new(adapter, offset))
    }
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> std::ops::Deref
    for LessThanExecutor<A, NUM_LIMBS, LIMB_BITS>
{
    type Target = LessThanExecutorInner<A, NUM_LIMBS, LIMB_BITS>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for LessThanExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    LessThanExecutorInner<A, NUM_LIMBS, LIMB_BITS>: PreflightExecutor<F, RA>,
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
pub(super) struct LessThanPreCompute {
    /// Second operand value (if immediate) or register index (if register)
    c: u32,
    /// Result register index
    a: u8,
    /// First operand register index
    b: u8,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> LessThanExecutor<A, NUM_LIMBS, LIMB_BITS> {
    /// Return `(is_imm, is_sltu)`.
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut LessThanPreCompute,
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
        if (d.as_canonical_u32() != RV32_REGISTER_AS)
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        // LessThan64Opcode has the same variant order as LessThanOpcode
        let local_opcode = LessThanOpcode::from_usize(opcode.local_opcode_idx(self.0.offset));
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = LessThanPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes::<{ RV32_REGISTER_NUM_LIMBS }>(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok((is_imm, local_opcode == LessThanOpcode::SLTU))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $is_sltu:ident, $num_limbs:expr) => {
        match ($is_imm, $is_sltu) {
            (true, true) => Ok($execute_impl::<_, _, true, true, $num_limbs>),
            (true, false) => Ok($execute_impl::<_, _, true, false, $num_limbs>),
            (false, true) => Ok($execute_impl::<_, _, false, true, $num_limbs>),
            (false, false) => Ok($execute_impl::<_, _, false, false, $num_limbs>),
        }
    };
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for LessThanExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<LessThanPreCompute>()
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
        let data: &mut LessThanPreCompute = data.borrow_mut();
        let (is_imm, is_sltu) = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, is_imm, is_sltu, NUM_LIMBS)
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for LessThanExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<LessThanPreCompute>>()
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
        let data: &mut E2PreCompute<LessThanPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, is_sltu) = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, is_imm, is_sltu, NUM_LIMBS)
    }
}

#[inline(always)]
fn to_u64<const N: usize>(bytes: [u8; N]) -> u64 {
    u64::from_le_bytes(std::array::from_fn(|i| if i < N { bytes[i] } else { 0 }))
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_U32: bool,
    const NUM_LIMBS: usize,
>(
    pre_compute: &LessThanPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    const { assert!(NUM_LIMBS == 4 || NUM_LIMBS == 8) };

    let fp = exec_state.memory.fp::<F>();
    let rs1 = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + (pre_compute.b as u32));
    let a = to_u64(rs1);
    let b = if E_IS_IMM {
        // pre_compute.c is already sign-extended to 32 bits by pre_compute_impl
        if NUM_LIMBS == 4 {
            pre_compute.c as u64
        } else {
            pre_compute.c as i32 as i64 as u64
        }
    } else {
        let rs2 = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + pre_compute.c);
        to_u64(rs2)
    };
    let cmp_result = if IS_U32 {
        a < b
    } else if NUM_LIMBS == 4 {
        (a as i32) < (b as i32)
    } else {
        (a as i64) < (b as i64)
    };
    let mut rd = [0u8; NUM_LIMBS];
    rd[0] = cmp_result as u8;
    exec_state.vm_write(RV32_REGISTER_AS, fp + (pre_compute.a as u32), &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_U32: bool,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &LessThanPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<LessThanPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, E_IS_IMM, IS_U32, NUM_LIMBS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_U32: bool,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<LessThanPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<LessThanPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, E_IS_IMM, IS_U32, NUM_LIMBS>(&pre_compute.data, exec_state);
    }
}

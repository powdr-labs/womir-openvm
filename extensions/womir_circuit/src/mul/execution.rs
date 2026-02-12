//! This module is largely a copy of extensions/rv32im/circuit/src/mul/execution.rs.
//! The only differences are:
//! - OpenVM's `MultiplicationExecutor` is wrapped to allow for trait implementations.
//! - In `execute_e12_impl`, we retrieve and add the frame pointer.
//! - Immediate operands are supported.
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
use openvm_rv32im_circuit::MultiplicationExecutor as MultiplicationExecutorInner;
use openvm_rv32im_transpiler::MulOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::{RV32_REGISTER_NUM_LIMBS, imm_to_bytes};

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, Copy)]
pub struct MultiplicationExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    pub MultiplicationExecutorInner<A, NUM_LIMBS, LIMB_BITS>,
);

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    MultiplicationExecutor<A, NUM_LIMBS, LIMB_BITS>
{
    pub fn new(adapter: A, offset: usize) -> Self {
        Self(MultiplicationExecutorInner::new(adapter, offset))
    }
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> std::ops::Deref
    for MultiplicationExecutor<A, NUM_LIMBS, LIMB_BITS>
{
    type Target = MultiplicationExecutorInner<A, NUM_LIMBS, LIMB_BITS>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for MultiplicationExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    MultiplicationExecutorInner<A, NUM_LIMBS, LIMB_BITS>: PreflightExecutor<F, RA>,
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
pub(super) struct MulPreCompute {
    /// Second operand value (if immediate) or register index (if register)
    c: u32,
    /// Result register index
    a: u8,
    /// First operand register index
    b: u8,
    /// Whether the second operand is an immediate
    is_imm: bool,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    MultiplicationExecutor<A, NUM_LIMBS, LIMB_BITS>
{
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut MulPreCompute,
    ) -> Result<(), StaticProgramError> {
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
        debug_assert_eq!(
            MulOpcode::from_usize(opcode.local_opcode_idx(self.0.offset)),
            MulOpcode::MUL
        );
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = MulPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes::<{ RV32_REGISTER_NUM_LIMBS }>(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            is_imm,
        };
        Ok(())
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for MultiplicationExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<MulPreCompute>()
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
        let data: &mut MulPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;

        if data.is_imm {
            Ok(execute_e1_handler::<_, _, true, NUM_LIMBS>)
        } else {
            Ok(execute_e1_handler::<_, _, false, NUM_LIMBS>)
        }
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for MultiplicationExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<MulPreCompute>>()
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
        let data: &mut E2PreCompute<MulPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;

        if data.data.is_imm {
            Ok(execute_e2_handler::<_, _, true, NUM_LIMBS>)
        } else {
            Ok(execute_e2_handler::<_, _, false, NUM_LIMBS>)
        }
    }
}

/// Sign-extend a u32 value to `[u8; N]`.
/// For N=4, this is equivalent to `c.to_le_bytes()`.
/// For N>4, the upper bytes are sign-extended from bit 31.
#[inline(always)]
fn sign_extend_u32<const N: usize>(c: u32) -> [u8; N] {
    let sign_byte = if c & 0x8000_0000 != 0 { 0xFF } else { 0x00 };
    let le = c.to_le_bytes();
    std::array::from_fn(|i| if i < 4 { le[i] } else { sign_byte })
}

/// Wrapping multiplication of two byte arrays, returning the lower NUM_LIMBS bytes.
#[inline(always)]
fn wrapping_mul_bytes<const NUM_LIMBS: usize>(
    a: &[u8; NUM_LIMBS],
    b: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    let mut result = [0u8; NUM_LIMBS];
    let mut carry = 0u32;
    for i in 0..NUM_LIMBS {
        let mut sum = carry;
        for j in 0..=i {
            sum += (a[j] as u32) * (b[i - j] as u32);
        }
        result[i] = (sum & 0xFF) as u8;
        carry = sum >> 8;
    }
    result
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const NUM_LIMBS: usize,
>(
    pre_compute: &MulPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let fp = exec_state.memory.fp::<F>();
    let rs1 = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + (pre_compute.b as u32));
    let rs2 = if E_IS_IMM {
        sign_extend_u32::<NUM_LIMBS>(pre_compute.c)
    } else {
        exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + pre_compute.c)
    };

    let rd = wrapping_mul_bytes::<NUM_LIMBS>(&rs1, &rs2);

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
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &MulPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<MulPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, E_IS_IMM, NUM_LIMBS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const E_IS_IMM: bool,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<MulPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<MulPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, E_IS_IMM, NUM_LIMBS>(&pre_compute.data, exec_state);
    }
}

//! This module is largely a copy of extensions/rv32im/circuit/src/divrem/execution.rs.
//! The only differences are:
//! - OpenVM's `DivRemExecutor` is wrapped to allow for trait implementations.
//! - In `execute_e12_impl`, we retrieve and add the frame pointer.
//! - Operations are generic over NUM_LIMBS to support both 32-bit and 64-bit.
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
    LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_rv32im_circuit::DivRemExecutor as DivRemExecutorInner;
use openvm_rv32im_transpiler::DivRemOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, Copy)]
pub struct DivRemExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    pub DivRemExecutorInner<A, NUM_LIMBS, LIMB_BITS>,
);

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> DivRemExecutor<A, NUM_LIMBS, LIMB_BITS> {
    pub fn new(adapter: A, offset: usize) -> Self {
        Self(DivRemExecutorInner::new(adapter, offset))
    }
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> std::ops::Deref
    for DivRemExecutor<A, NUM_LIMBS, LIMB_BITS>
{
    type Target = DivRemExecutorInner<A, NUM_LIMBS, LIMB_BITS>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for DivRemExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    DivRemExecutorInner<A, NUM_LIMBS, LIMB_BITS>: PreflightExecutor<F, RA>,
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
pub(super) struct DivRemPreCompute {
    /// Result register index
    a: u16,
    /// First operand register index
    b: u16,
    /// Second operand register index
    c: u16,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> DivRemExecutor<A, NUM_LIMBS, LIMB_BITS> {
    /// Return the local opcode.
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut DivRemPreCompute,
    ) -> Result<DivRemOpcode, StaticProgramError> {
        let Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let local_opcode = DivRemOpcode::from_usize(opcode.local_opcode_idx(self.0.offset));
        *data = DivRemPreCompute {
            a: a.as_canonical_u32() as u16,
            b: b.as_canonical_u32() as u16,
            c: c.as_canonical_u32() as u16,
        };
        Ok(local_opcode)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident, $num_limbs:expr) => {
        match $local_opcode {
            DivRemOpcode::DIV => Ok($execute_impl::<_, _, { DivRemOpcode::DIV as u8 }, $num_limbs>),
            DivRemOpcode::DIVU => {
                Ok($execute_impl::<_, _, { DivRemOpcode::DIVU as u8 }, $num_limbs>)
            }
            DivRemOpcode::REM => Ok($execute_impl::<_, _, { DivRemOpcode::REM as u8 }, $num_limbs>),
            DivRemOpcode::REMU => {
                Ok($execute_impl::<_, _, { DivRemOpcode::REMU as u8 }, $num_limbs>)
            }
        }
    };
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for DivRemExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<DivRemPreCompute>()
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
        let data: &mut DivRemPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, local_opcode, NUM_LIMBS)
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for DivRemExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<DivRemPreCompute>>()
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
        let data: &mut E2PreCompute<DivRemPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, local_opcode, NUM_LIMBS)
    }
}

/// Two's complement negate of a byte array.
#[inline(always)]
fn negate_bytes<const N: usize>(x: &[u8; N]) -> [u8; N] {
    let mut carry = 1u16;
    std::array::from_fn(|i| {
        let val = (!x[i]) as u16 + carry;
        carry = val >> 8;
        (val & 0xFF) as u8
    })
}

/// Check if a two's complement byte array is negative (MSB of last byte set).
#[inline(always)]
fn is_negative<const N: usize>(x: &[u8; N]) -> bool {
    x[N - 1] & 0x80 != 0
}

/// Check if a byte array is zero.
#[inline(always)]
fn is_zero<const N: usize>(x: &[u8; N]) -> bool {
    x.iter().all(|&b| b == 0)
}

/// Check if a byte array equals the minimum signed value (0x80, 0x00, ..., 0x00).
#[inline(always)]
fn is_signed_min<const N: usize>(x: &[u8; N]) -> bool {
    x[N - 1] == 0x80 && x[..N - 1].iter().all(|&b| b == 0)
}

/// Unsigned division of byte arrays using schoolbook algorithm.
/// Returns (quotient, remainder).
#[inline(always)]
fn unsigned_divrem<const N: usize>(a: &[u8; N], b: &[u8; N]) -> ([u8; N], [u8; N]) {
    // Simple implementation: convert to BigUint
    use num_bigint::BigUint;

    let a_big = BigUint::from_bytes_le(a);
    let b_big = BigUint::from_bytes_le(b);

    let q_big = &a_big / &b_big;
    let r_big = &a_big % &b_big;

    let mut q = [0u8; N];
    let q_bytes = q_big.to_bytes_le();
    q[..q_bytes.len().min(N)].copy_from_slice(&q_bytes[..q_bytes.len().min(N)]);

    let mut r = [0u8; N];
    let r_bytes = r_big.to_bytes_le();
    r[..r_bytes.len().min(N)].copy_from_slice(&r_bytes[..r_bytes.len().min(N)]);

    (q, r)
}

/// Compute DivRem result for a given opcode.
/// RISC-V semantics: division by zero returns all-ones for div, dividend for rem.
/// Signed overflow (MIN / -1) returns MIN for div, 0 for rem.
#[inline(always)]
fn compute_divrem<const OPCODE: u8, const NUM_LIMBS: usize>(
    rs1: &[u8; NUM_LIMBS],
    rs2: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    match OPCODE {
        x if x == DivRemOpcode::DIV as u8 => {
            // Signed division
            if is_zero(rs2) {
                return [0xFF; NUM_LIMBS]; // All ones
            }
            if is_signed_min(rs1) && rs2.iter().all(|&b| b == 0xFF) {
                return *rs1; // MIN / -1 = MIN (overflow)
            }
            let a_neg = is_negative(rs1);
            let b_neg = is_negative(rs2);
            let a_abs = if a_neg { negate_bytes(rs1) } else { *rs1 };
            let b_abs = if b_neg { negate_bytes(rs2) } else { *rs2 };
            let (q, _) = unsigned_divrem(&a_abs, &b_abs);
            if a_neg ^ b_neg { negate_bytes(&q) } else { q }
        }
        x if x == DivRemOpcode::DIVU as u8 => {
            // Unsigned division
            if is_zero(rs2) {
                return [0xFF; NUM_LIMBS]; // All ones
            }
            let (q, _) = unsigned_divrem(rs1, rs2);
            q
        }
        x if x == DivRemOpcode::REM as u8 => {
            // Signed remainder
            if is_zero(rs2) {
                return *rs1; // Dividend
            }
            if is_signed_min(rs1) && rs2.iter().all(|&b| b == 0xFF) {
                return [0; NUM_LIMBS]; // MIN % -1 = 0
            }
            let a_neg = is_negative(rs1);
            let b_neg = is_negative(rs2);
            let a_abs = if a_neg { negate_bytes(rs1) } else { *rs1 };
            let b_abs = if b_neg { negate_bytes(rs2) } else { *rs2 };
            let (_, r) = unsigned_divrem(&a_abs, &b_abs);
            // Remainder has sign of dividend
            if a_neg { negate_bytes(&r) } else { r }
        }
        x if x == DivRemOpcode::REMU as u8 => {
            // Unsigned remainder
            if is_zero(rs2) {
                return *rs1; // Dividend
            }
            let (_, r) = unsigned_divrem(rs1, rs2);
            r
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const OPCODE: u8,
    const NUM_LIMBS: usize,
>(
    pre_compute: &DivRemPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let fp = exec_state.memory.fp::<F>();
    let rs1 = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + (pre_compute.b as u32));
    let rs2 = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + (pre_compute.c as u32));

    let rd = compute_divrem::<OPCODE, NUM_LIMBS>(&rs1, &rs2);

    exec_state.vm_write(RV32_REGISTER_AS, fp + (pre_compute.a as u32), &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const OPCODE: u8,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &DivRemPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<DivRemPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, OPCODE, NUM_LIMBS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const OPCODE: u8,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<DivRemPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<DivRemPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, OPCODE, NUM_LIMBS>(&pre_compute.data, exec_state);
    }
}

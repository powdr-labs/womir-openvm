//! This module is largely a copy of extensions/rv32im/circuit/src/base_alu/execution.rs.
//! The only differences are:
//! - OpenVM's `BaseAluExecutor` is wrapped to allow for trait implementations.
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
use openvm_rv32im_circuit::BaseAluExecutor as BaseAluExecutorInner;
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[allow(unused_imports)]
use crate::adapters::imm_to_bytes;

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, Copy)]
pub struct BaseAluExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    pub BaseAluExecutorInner<A, NUM_LIMBS, LIMB_BITS>,
);

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAluExecutor<A, NUM_LIMBS, LIMB_BITS> {
    pub fn new(adapter: A, offset: usize) -> Self {
        Self(BaseAluExecutorInner::new(adapter, offset))
    }
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> std::ops::Deref
    for BaseAluExecutor<A, NUM_LIMBS, LIMB_BITS>
{
    type Target = BaseAluExecutorInner<A, NUM_LIMBS, LIMB_BITS>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for BaseAluExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    BaseAluExecutorInner<A, NUM_LIMBS, LIMB_BITS>: PreflightExecutor<F, RA>,
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
pub(super) struct BaseAluPreCompute {
    /// Second operand value (if immediate) or register index (if register)
    c: u32,
    /// Result register index
    a: u8,
    /// First operand register index
    b: u8,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAluExecutor<A, NUM_LIMBS, LIMB_BITS> {
    /// Return `is_imm`, true if `e` is RV32_IMM_AS.
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BaseAluPreCompute,
    ) -> Result<bool, StaticProgramError> {
        let Instruction { a, b, c, d, e, .. } = inst;
        let e_u32 = e.as_canonical_u32();
        if (d.as_canonical_u32() != RV32_REGISTER_AS)
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = BaseAluPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok(is_imm)
    }
}

// ==================== dispatch ====================

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $opcode:expr, $offset:expr, $num_limbs:expr) => {
        Ok(
            // BaseAlu64Opcode has the same variant order as BaseAluOpcode
            match (
                $is_imm,
                BaseAluOpcode::from_usize($opcode.local_opcode_idx($offset)),
            ) {
                (true, BaseAluOpcode::ADD) => $execute_impl::<_, _, true, AddOp, $num_limbs>,
                (false, BaseAluOpcode::ADD) => $execute_impl::<_, _, false, AddOp, $num_limbs>,
                (true, BaseAluOpcode::SUB) => $execute_impl::<_, _, true, SubOp, $num_limbs>,
                (false, BaseAluOpcode::SUB) => $execute_impl::<_, _, false, SubOp, $num_limbs>,
                (true, BaseAluOpcode::XOR) => $execute_impl::<_, _, true, XorOp, $num_limbs>,
                (false, BaseAluOpcode::XOR) => $execute_impl::<_, _, false, XorOp, $num_limbs>,
                (true, BaseAluOpcode::OR) => $execute_impl::<_, _, true, OrOp, $num_limbs>,
                (false, BaseAluOpcode::OR) => $execute_impl::<_, _, false, OrOp, $num_limbs>,
                (true, BaseAluOpcode::AND) => $execute_impl::<_, _, true, AndOp, $num_limbs>,
                (false, BaseAluOpcode::AND) => $execute_impl::<_, _, false, AndOp, $num_limbs>,
            },
        )
    };
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for BaseAluExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BaseAluPreCompute>()
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
        let data: &mut BaseAluPreCompute = data.borrow_mut();
        let is_imm = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(
            execute_e1_handler,
            is_imm,
            inst.opcode,
            self.0.offset,
            NUM_LIMBS
        )
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for BaseAluExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BaseAluPreCompute>>()
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
        let data: &mut E2PreCompute<BaseAluPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_imm = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(
            execute_e2_handler,
            is_imm,
            inst.opcode,
            self.0.offset,
            NUM_LIMBS
        )
    }
}

// ==================== execute ====================

/// Sign-extend a u32 value to `[u8; N]`.
/// For N=4, this is equivalent to `c.to_le_bytes()`.
/// For N>4, the upper bytes are sign-extended from bit 31.
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
    OP: AluOp,
    const NUM_LIMBS: usize,
>(
    pre_compute: &BaseAluPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    // NUM_LIMBS must be a multiple of 4 since we process data in 4-byte (u32) chunks
    const {
        assert!(
            NUM_LIMBS.is_multiple_of(4),
            "NUM_LIMBS must be a multiple of 4"
        )
    };

    let fp = exec_state.memory.fp::<F>();
    let rs1 = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + (pre_compute.b as u32));
    let rs2 = if IS_IMM {
        sign_extend_u32::<NUM_LIMBS>(pre_compute.c)
    } else {
        exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + pre_compute.c)
    };
    let mut rd = [0u8; NUM_LIMBS];
    let mut carry = 0u32;
    for w in 0..NUM_LIMBS / 4 {
        let i = w * 4;
        let a = u32::from_le_bytes(rs1[i..i + 4].try_into().unwrap());
        let b = u32::from_le_bytes(rs2[i..i + 4].try_into().unwrap());
        let (result, new_carry) = OP::compute(a, b, carry);
        carry = new_carry;
        rd[i..i + 4].copy_from_slice(&result.to_le_bytes());
    }
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
    OP: AluOp,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &BaseAluPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<BaseAluPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, IS_IMM, OP, NUM_LIMBS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<BaseAluPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<BaseAluPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, IS_IMM, OP, NUM_LIMBS>(&pre_compute.data, exec_state);
    }
}

// ==================== ALU operation trait ====================

trait AluOp {
    /// Compute the operation on a single u32 word with carry/borrow propagation.
    /// Returns (result, carry_out).
    fn compute(a: u32, b: u32, carry: u32) -> (u32, u32);
}

struct AddOp;
struct SubOp;
struct XorOp;
struct OrOp;
struct AndOp;

impl AluOp for AddOp {
    #[inline(always)]
    fn compute(a: u32, b: u32, carry: u32) -> (u32, u32) {
        let sum = a as u64 + b as u64 + carry as u64;
        (sum as u32, (sum >> 32) as u32)
    }
}
impl AluOp for SubOp {
    #[inline(always)]
    fn compute(a: u32, b: u32, borrow: u32) -> (u32, u32) {
        let diff = a as u64 as i64 - b as u64 as i64 - borrow as i64;
        (diff as u32, if diff < 0 { 1 } else { 0 })
    }
}
impl AluOp for XorOp {
    #[inline(always)]
    fn compute(a: u32, b: u32, _: u32) -> (u32, u32) {
        (a ^ b, 0)
    }
}
impl AluOp for OrOp {
    #[inline(always)]
    fn compute(a: u32, b: u32, _: u32) -> (u32, u32) {
        (a | b, 0)
    }
}
impl AluOp for AndOp {
    #[inline(always)]
    fn compute(a: u32, b: u32, _: u32) -> (u32, u32) {
        (a & b, 0)
    }
}

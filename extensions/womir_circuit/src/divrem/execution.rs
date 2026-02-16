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
    a: u32,
    /// First operand register index
    b: u32,
    /// Second operand register index
    c: u32,
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
            a: a.as_canonical_u32(),
            b: b.as_canonical_u32(),
            c: c.as_canonical_u32(),
        };
        Ok(local_opcode)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident, $num_limbs:expr) => {
        match $local_opcode {
            DivRemOpcode::DIV => Ok($execute_impl::<_, _, DivOp, $num_limbs>),
            DivRemOpcode::DIVU => Ok($execute_impl::<_, _, DivuOp, $num_limbs>),
            DivRemOpcode::REM => Ok($execute_impl::<_, _, RemOp, $num_limbs>),
            DivRemOpcode::REMU => Ok($execute_impl::<_, _, RemuOp, $num_limbs>),
        }
    };
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for DivRemExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    DivOp: DivRemOp<NUM_LIMBS>,
    DivuOp: DivRemOp<NUM_LIMBS>,
    RemOp: DivRemOp<NUM_LIMBS>,
    RemuOp: DivRemOp<NUM_LIMBS>,
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
    DivOp: DivRemOp<NUM_LIMBS>,
    DivuOp: DivRemOp<NUM_LIMBS>,
    RemOp: DivRemOp<NUM_LIMBS>,
    RemuOp: DivRemOp<NUM_LIMBS>,
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

// ==================== DivRem operation trait ====================

trait DivRemOp<const NUM_LIMBS: usize> {
    fn compute(rs1: [u8; NUM_LIMBS], rs2: [u8; NUM_LIMBS]) -> [u8; NUM_LIMBS];
}

struct DivOp;
struct DivuOp;
struct RemOp;
struct RemuOp;

impl DivRemOp<4> for DivOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1 = i32::from_le_bytes(rs1);
        let rs2 = i32::from_le_bytes(rs2);
        match (rs1, rs2) {
            (_, 0) => [u8::MAX; 4],
            (i32::MIN, -1) => i32::MIN.to_le_bytes(),
            _ => (rs1 / rs2).to_le_bytes(),
        }
    }
}

impl DivRemOp<8> for DivOp {
    #[inline(always)]
    fn compute(rs1: [u8; 8], rs2: [u8; 8]) -> [u8; 8] {
        let rs1 = i64::from_le_bytes(rs1);
        let rs2 = i64::from_le_bytes(rs2);
        match (rs1, rs2) {
            (_, 0) => [u8::MAX; 8],
            (i64::MIN, -1) => i64::MIN.to_le_bytes(),
            _ => (rs1 / rs2).to_le_bytes(),
        }
    }
}

impl DivRemOp<4> for DivuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1 = u32::from_le_bytes(rs1);
        let rs2 = u32::from_le_bytes(rs2);
        match rs2 {
            0 => [u8::MAX; 4],
            _ => (rs1 / rs2).to_le_bytes(),
        }
    }
}

impl DivRemOp<8> for DivuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 8], rs2: [u8; 8]) -> [u8; 8] {
        let rs1 = u64::from_le_bytes(rs1);
        let rs2 = u64::from_le_bytes(rs2);
        match rs2 {
            0 => [u8::MAX; 8],
            _ => (rs1 / rs2).to_le_bytes(),
        }
    }
}

impl DivRemOp<4> for RemOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1 = i32::from_le_bytes(rs1);
        let rs2 = i32::from_le_bytes(rs2);
        match (rs1, rs2) {
            (_, 0) => rs1.to_le_bytes(),
            (i32::MIN, -1) => 0i32.to_le_bytes(),
            _ => (rs1 % rs2).to_le_bytes(),
        }
    }
}

impl DivRemOp<8> for RemOp {
    #[inline(always)]
    fn compute(rs1: [u8; 8], rs2: [u8; 8]) -> [u8; 8] {
        let rs1 = i64::from_le_bytes(rs1);
        let rs2 = i64::from_le_bytes(rs2);
        match (rs1, rs2) {
            (_, 0) => rs1.to_le_bytes(),
            (i64::MIN, -1) => 0i64.to_le_bytes(),
            _ => (rs1 % rs2).to_le_bytes(),
        }
    }
}

impl DivRemOp<4> for RemuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1 = u32::from_le_bytes(rs1);
        let rs2 = u32::from_le_bytes(rs2);
        match rs2 {
            0 => rs1.to_le_bytes(),
            _ => (rs1 % rs2).to_le_bytes(),
        }
    }
}

impl DivRemOp<8> for RemuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 8], rs2: [u8; 8]) -> [u8; 8] {
        let rs1 = u64::from_le_bytes(rs1);
        let rs2 = u64::from_le_bytes(rs2);
        match rs2 {
            0 => rs1.to_le_bytes(),
            _ => (rs1 % rs2).to_le_bytes(),
        }
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    OP: DivRemOp<NUM_LIMBS>,
    const NUM_LIMBS: usize,
>(
    pre_compute: &DivRemPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let fp = exec_state.memory.fp::<F>();
    let rs1 = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + pre_compute.b);
    let rs2 = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + pre_compute.c);

    let rd = OP::compute(rs1, rs2);

    exec_state.vm_write(RV32_REGISTER_AS, fp + pre_compute.a, &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    OP: DivRemOp<NUM_LIMBS>,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &DivRemPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<DivRemPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, OP, NUM_LIMBS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    OP: DivRemOp<NUM_LIMBS>,
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
        execute_e12_impl::<F, CTX, OP, NUM_LIMBS>(&pre_compute.data, exec_state);
    }
}

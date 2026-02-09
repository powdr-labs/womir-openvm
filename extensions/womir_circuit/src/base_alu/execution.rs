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
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_rv32im_circuit::BaseAluExecutor as BaseAluExecutorInner;
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[allow(unused_imports)]
use crate::{adapters::imm_to_bytes, common::*};

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
    c: u32,
    a: u8,
    b: u8,
}

impl<A, const LIMB_BITS: usize> BaseAluExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> {
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

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $opcode:expr, $offset:expr) => {
        Ok(
            match (
                $is_imm,
                BaseAluOpcode::from_usize($opcode.local_opcode_idx($offset)),
            ) {
                (true, BaseAluOpcode::ADD) => $execute_impl::<_, _, true, AddOp>,
                (false, BaseAluOpcode::ADD) => $execute_impl::<_, _, false, AddOp>,
                (true, BaseAluOpcode::SUB) => $execute_impl::<_, _, true, SubOp>,
                (false, BaseAluOpcode::SUB) => $execute_impl::<_, _, false, SubOp>,
                (true, BaseAluOpcode::XOR) => $execute_impl::<_, _, true, XorOp>,
                (false, BaseAluOpcode::XOR) => $execute_impl::<_, _, false, XorOp>,
                (true, BaseAluOpcode::OR) => $execute_impl::<_, _, true, OrOp>,
                (false, BaseAluOpcode::OR) => $execute_impl::<_, _, false, OrOp>,
                (true, BaseAluOpcode::AND) => $execute_impl::<_, _, true, AndOp>,
                (false, BaseAluOpcode::AND) => $execute_impl::<_, _, false, AndOp>,
            },
        )
    };
}

impl<F, A, const LIMB_BITS: usize> InterpreterExecutor<F>
    for BaseAluExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
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

        dispatch!(execute_e1_handler, is_imm, inst.opcode, self.0.offset)
    }
}

impl<F, A, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for BaseAluExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
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

        dispatch!(execute_e2_handler, is_imm, inst.opcode, self.0.offset)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp,
>(
    pre_compute: &BaseAluPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let fp = exec_state.memory.fp();
    let rs1 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, fp + (pre_compute.b as u32));
    let rs2 = if IS_IMM {
        pre_compute.c.to_le_bytes()
    } else {
        exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, fp + pre_compute.c)
    };
    let rs1 = u32::from_le_bytes(rs1);
    let rs2 = u32::from_le_bytes(rs2);
    let rd = <OP as AluOp>::compute(rs1, rs2);
    let rd = rd.to_le_bytes();
    exec_state.vm_write::<u8, 4>(RV32_REGISTER_AS, fp + (pre_compute.a as u32), &rd);
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
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &BaseAluPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<BaseAluPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, IS_IMM, OP>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp,
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
        execute_e12_impl::<F, CTX, IS_IMM, OP>(&pre_compute.data, exec_state);
    }
}

trait AluOp {
    fn compute(rs1: u32, rs2: u32) -> u32;
}
struct AddOp;
struct SubOp;
struct XorOp;
struct OrOp;
struct AndOp;
impl AluOp for AddOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1.wrapping_add(rs2)
    }
}
impl AluOp for SubOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1.wrapping_sub(rs2)
    }
}
impl AluOp for XorOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1 ^ rs2
    }
}
impl AluOp for OrOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1 | rs2
    }
}
impl AluOp for AndOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1 & rs2
    }
}

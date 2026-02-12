//! This module is largely a copy of extensions/rv32im/circuit/src/shift/execution.rs.
//! The only differences are:
//! - OpenVM's `ShiftExecutor` is wrapped to allow for trait implementations.
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
use openvm_rv32im_circuit::ShiftExecutor as ShiftExecutorInner;
use openvm_rv32im_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::{RV32_REGISTER_NUM_LIMBS, imm_to_bytes};

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, Copy)]
pub struct ShiftExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    pub ShiftExecutorInner<A, NUM_LIMBS, LIMB_BITS>,
);

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> ShiftExecutor<A, NUM_LIMBS, LIMB_BITS> {
    pub fn new(adapter: A, offset: usize) -> Self {
        Self(ShiftExecutorInner::new(adapter, offset))
    }
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> std::ops::Deref
    for ShiftExecutor<A, NUM_LIMBS, LIMB_BITS>
{
    type Target = ShiftExecutorInner<A, NUM_LIMBS, LIMB_BITS>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for ShiftExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    ShiftExecutorInner<A, NUM_LIMBS, LIMB_BITS>: PreflightExecutor<F, RA>,
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
pub(super) struct ShiftPreCompute {
    /// Second operand value (if immediate) or register index (if register)
    c: u32,
    /// Result register index
    a: u8,
    /// First operand register index
    b: u8,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> ShiftExecutor<A, NUM_LIMBS, LIMB_BITS> {
    /// Return `(is_imm, opcode)`.
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ShiftPreCompute,
    ) -> Result<(bool, ShiftOpcode), StaticProgramError> {
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
        // Shift64Opcode has the same variant order as ShiftOpcode
        let local_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.0.offset));
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = ShiftPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes::<{ RV32_REGISTER_NUM_LIMBS }>(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok((is_imm, local_opcode))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $opcode:ident, $num_limbs:expr) => {
        match ($is_imm, $opcode) {
            (true, ShiftOpcode::SLL) => {
                Ok($execute_impl::<_, _, true, { ShiftOpcode::SLL as u8 }, $num_limbs>)
            }
            (false, ShiftOpcode::SLL) => {
                Ok($execute_impl::<_, _, false, { ShiftOpcode::SLL as u8 }, $num_limbs>)
            }
            (true, ShiftOpcode::SRL) => {
                Ok($execute_impl::<_, _, true, { ShiftOpcode::SRL as u8 }, $num_limbs>)
            }
            (false, ShiftOpcode::SRL) => {
                Ok($execute_impl::<_, _, false, { ShiftOpcode::SRL as u8 }, $num_limbs>)
            }
            (true, ShiftOpcode::SRA) => {
                Ok($execute_impl::<_, _, true, { ShiftOpcode::SRA as u8 }, $num_limbs>)
            }
            (false, ShiftOpcode::SRA) => {
                Ok($execute_impl::<_, _, false, { ShiftOpcode::SRA as u8 }, $num_limbs>)
            }
        }
    };
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for ShiftExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftPreCompute>()
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
        let data: &mut ShiftPreCompute = data.borrow_mut();
        let (is_imm, opcode) = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, is_imm, opcode, NUM_LIMBS)
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for ShiftExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShiftPreCompute>>()
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
        let data: &mut E2PreCompute<ShiftPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, is_imm, opcode, NUM_LIMBS)
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

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const OPCODE: u8,
    const NUM_LIMBS: usize,
>(
    pre_compute: &ShiftPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let num_bits = NUM_LIMBS * 8;
    let fp = exec_state.memory.fp::<F>();
    let rs1 = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + (pre_compute.b as u32));
    let rs2 = if E_IS_IMM {
        sign_extend_u32::<NUM_LIMBS>(pre_compute.c)
    } else {
        exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + pre_compute.c)
    };

    // Shift amount is taken mod num_bits, from the lowest byte only
    let shift_amount = (rs2[0] as usize) % num_bits;

    let rd = match OPCODE {
        x if x == ShiftOpcode::SLL as u8 => {
            // Shift left logical
            shift_left::<NUM_LIMBS>(&rs1, shift_amount)
        }
        x if x == ShiftOpcode::SRL as u8 => {
            // Shift right logical
            shift_right::<NUM_LIMBS>(&rs1, shift_amount, false)
        }
        x if x == ShiftOpcode::SRA as u8 => {
            // Shift right arithmetic
            shift_right::<NUM_LIMBS>(&rs1, shift_amount, true)
        }
        _ => unreachable!(),
    };

    exec_state.vm_write(RV32_REGISTER_AS, fp + (pre_compute.a as u32), &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[inline(always)]
fn shift_left<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], shift: usize) -> [u8; NUM_LIMBS] {
    let limb_shift = shift / 8;
    let bit_shift = shift % 8;
    let mut result = [0u8; NUM_LIMBS];
    for i in limb_shift..NUM_LIMBS {
        result[i] = if i > limb_shift {
            (((x[i - limb_shift] as u16) << bit_shift)
                | ((x[i - limb_shift - 1] as u16) >> (8 - bit_shift)))
                % 256
        } else {
            ((x[i - limb_shift] as u16) << bit_shift) % 256
        } as u8;
    }
    result
}

#[inline(always)]
fn shift_right<const NUM_LIMBS: usize>(
    x: &[u8; NUM_LIMBS],
    shift: usize,
    arithmetic: bool,
) -> [u8; NUM_LIMBS] {
    let fill = if arithmetic {
        0xFF * (x[NUM_LIMBS - 1] >> 7)
    } else {
        0
    };
    let limb_shift = shift / 8;
    let bit_shift = shift % 8;
    let mut result = [fill; NUM_LIMBS];
    for i in 0..(NUM_LIMBS - limb_shift) {
        result[i] = if i + limb_shift + 1 < NUM_LIMBS {
            (((x[i + limb_shift] >> bit_shift) as u16)
                | ((x[i + limb_shift + 1] as u16) << (8 - bit_shift)))
                % 256
        } else {
            (((x[i + limb_shift] >> bit_shift) as u16) | ((fill as u16) << (8 - bit_shift))) % 256
        } as u8;
    }
    result
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const OPCODE: u8,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &ShiftPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<ShiftPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, E_IS_IMM, OPCODE, NUM_LIMBS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const E_IS_IMM: bool,
    const OPCODE: u8,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<ShiftPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<ShiftPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, E_IS_IMM, OPCODE, NUM_LIMBS>(&pre_compute.data, exec_state);
    }
}

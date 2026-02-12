use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::{
    HintStoreOpcode,
    HintStoreOpcode::{HINT_BUFFER, HINT_STOREW},
};

use super::Rv32HintStoreExecutor;
use crate::memory_config::FpMemory;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct HintStorePreCompute {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

impl Rv32HintStoreExecutor {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut HintStorePreCompute,
    ) -> Result<HintStoreOpcode, StaticProgramError> {
        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        *data = HintStorePreCompute {
            a: a.as_canonical_u32(),
            b: b.as_canonical_u32(),
            c: c.as_canonical_u32(),
            d: d.as_canonical_u32(),
        };
        Ok(HintStoreOpcode::from_usize(
            opcode.local_opcode_idx(self.offset),
        ))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        match $local_opcode {
            HINT_STOREW => Ok($execute_impl::<_, _, true>),
            HINT_BUFFER => Ok($execute_impl::<_, _, false>),
        }
    };
}

impl<F> InterpreterExecutor<F> for Rv32HintStoreExecutor
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<HintStorePreCompute>()
    }

    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut HintStorePreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F> InterpreterMeteredExecutor<F> for Rv32HintStoreExecutor
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<HintStorePreCompute>>()
    }

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
        let pre_compute: &mut E2PreCompute<HintStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

/// Return the number of used rows.
#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_HINT_STOREW: bool>(
    pre_compute: &HintStorePreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<u32, ExecutionError> {
    let pc = exec_state.pc();
    let fp = exec_state.memory.fp::<F>();

    if IS_HINT_STOREW {
        // STOREW: a=rd*4, b=0, c=0, d=RV32_REGISTER_AS
        // Write hint to (RV32_REGISTER_AS, a + fp)
        let num_words = 1u32;

        if exec_state.streams.hint_stream.len() < RV32_REGISTER_NUM_LIMBS {
            return Err(ExecutionError::HintOutOfBounds { pc });
        }

        let data: [u8; RV32_REGISTER_NUM_LIMBS] = std::array::from_fn(|_| {
            exec_state
                .streams
                .hint_stream
                .pop_front()
                .unwrap()
                .as_canonical_u32() as u8
        });
        exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a + fp, &data);

        exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
        Ok(num_words)
    } else {
        // BUFFER: a=0, b=num_words_reg*4, c=mem_ptr_reg*4, d=mem_imm
        let num_words_limbs = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b + fp);
        let num_words = u32::from_le_bytes(num_words_limbs);

        let mem_ptr_limbs = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c + fp);
        let mem_ptr = u32::from_le_bytes(mem_ptr_limbs);

        debug_assert_ne!(num_words, 0);

        if exec_state.streams.hint_stream.len() < RV32_REGISTER_NUM_LIMBS * num_words as usize {
            return Err(ExecutionError::HintOutOfBounds { pc });
        }

        for word_index in 0..num_words {
            let data: [u8; RV32_REGISTER_NUM_LIMBS] = std::array::from_fn(|_| {
                exec_state
                    .streams
                    .hint_stream
                    .pop_front()
                    .unwrap()
                    .as_canonical_u32() as u8
            });
            exec_state.vm_write(
                RV32_MEMORY_AS,
                mem_ptr + pre_compute.d + (RV32_REGISTER_NUM_LIMBS as u32 * word_index),
                &data,
            );
        }

        exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
        Ok(num_words)
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_HINT_STOREW: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &HintStorePreCompute = unsafe {
        std::slice::from_raw_parts(pre_compute, size_of::<HintStorePreCompute>()).borrow()
    };
    unsafe { execute_e12_impl::<F, CTX, IS_HINT_STOREW>(pre_compute, exec_state) }?;
    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_HINT_STOREW: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<HintStorePreCompute> = unsafe {
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<HintStorePreCompute>>())
            .borrow()
    };
    let height_delta =
        unsafe { execute_e12_impl::<F, CTX, IS_HINT_STOREW>(&pre_compute.data, exec_state) }?;
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height_delta);
    Ok(())
}

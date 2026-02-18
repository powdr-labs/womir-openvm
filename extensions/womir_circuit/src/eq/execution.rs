use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::{eq::core::EqCoreRecord, memory_config::FpMemory};
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::EqOpcode;

use crate::adapters::{
    BaseAluAdapterExecutorDifferentInputsOutputs, RV32_REGISTER_NUM_LIMBS, imm_to_bytes,
};

#[derive(Clone)]
pub struct EqExecutor<
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
    const NUM_WRITE_OPS: usize,
    const LIMB_BITS: usize,
> {
    pub adapter: BaseAluAdapterExecutorDifferentInputsOutputs<
        NUM_LIMBS,
        NUM_READ_OPS,
        NUM_WRITE_OPS,
        LIMB_BITS,
    >,
    pub offset: usize,
}

impl<
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
    const NUM_WRITE_OPS: usize,
    const LIMB_BITS: usize,
> EqExecutor<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS, LIMB_BITS>
{
    pub fn new(
        adapter: BaseAluAdapterExecutorDifferentInputsOutputs<
            NUM_LIMBS,
            NUM_READ_OPS,
            NUM_WRITE_OPS,
            LIMB_BITS,
        >,
        offset: usize,
    ) -> Self {
        Self { adapter, offset }
    }

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
        if (d.as_canonical_u32() != RV32_REGISTER_AS)
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = EqOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();

        *data = EqPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes::<{ RV32_REGISTER_NUM_LIMBS }>(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32(),
            b: b.as_canonical_u32(),
        };

        Ok((is_imm, local_opcode == EqOpcode::EQ))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct EqPreCompute {
    c: u32,
    a: u32,
    b: u32,
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $is_eq:ident, $num_limbs:expr) => {
        match ($is_imm, $is_eq) {
            (true, true) => Ok($execute_impl::<_, _, true, true, $num_limbs>),
            (true, false) => Ok($execute_impl::<_, _, true, false, $num_limbs>),
            (false, true) => Ok($execute_impl::<_, _, false, true, $num_limbs>),
            (false, false) => Ok($execute_impl::<_, _, false, false, $num_limbs>),
        }
    };
}

impl<
    F,
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
    const NUM_WRITE_OPS: usize,
    const LIMB_BITS: usize,
> InterpreterExecutor<F> for EqExecutor<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<EqPreCompute>()
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
        let data: &mut EqPreCompute = data.borrow_mut();
        let (is_imm, is_eq) = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, is_imm, is_eq, NUM_LIMBS)
    }
}

impl<
    F,
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
    const NUM_WRITE_OPS: usize,
    const LIMB_BITS: usize,
> InterpreterMeteredExecutor<F> for EqExecutor<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS, LIMB_BITS>
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
        let data: &mut E2PreCompute<EqPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, is_eq) = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, is_imm, is_eq, NUM_LIMBS)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_EQ: bool,
    const NUM_LIMBS: usize,
>(
    pre_compute: &EqPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    const { assert!(NUM_LIMBS == 4 || NUM_LIMBS == 8) };

    let fp = exec_state.memory.fp::<F>();
    let lhs = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + pre_compute.b);
    let lhs_u64 = u64::from_le_bytes(std::array::from_fn(
        |i| if i < NUM_LIMBS { lhs[i] } else { 0 },
    ));
    let rhs_u64 = if E_IS_IMM {
        if NUM_LIMBS == 4 {
            pre_compute.c as u64
        } else {
            pre_compute.c as i32 as i64 as u64
        }
    } else {
        let rhs = exec_state.vm_read::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + pre_compute.c);
        u64::from_le_bytes(std::array::from_fn(
            |i| if i < NUM_LIMBS { rhs[i] } else { 0 },
        ))
    };

    let cmp_result = if IS_EQ {
        lhs_u64 == rhs_u64
    } else {
        lhs_u64 != rhs_u64
    };

    // Write only one register-width (4 bytes): comparison results are always i32,
    // even for 64-bit operands.
    let mut rd = [0u8; RV32_REGISTER_NUM_LIMBS];
    rd[0] = cmp_result as u8;
    exec_state.vm_write(RV32_REGISTER_AS, fp + pre_compute.a, &rd);

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_EQ: bool,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &EqPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<EqPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, E_IS_IMM, IS_EQ, NUM_LIMBS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_EQ: bool,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<EqPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<EqPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, E_IS_IMM, IS_EQ, NUM_LIMBS>(&pre_compute.data, exec_state);
    }
}

impl<
    F,
    RA,
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
    const NUM_WRITE_OPS: usize,
    const LIMB_BITS: usize,
> PreflightExecutor<F, RA> for EqExecutor<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS, LIMB_BITS>
where
    F: PrimeField32,
    BaseAluAdapterExecutorDifferentInputsOutputs<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS, LIMB_BITS>:
        AdapterTraceExecutor<
                F,
                ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
                WriteData: From<[[u8; NUM_LIMBS]; 1]>,
            >,
    for<'buf> RA: RecordArena<
            'buf,
            EmptyAdapterCoreLayout<
                F,
                BaseAluAdapterExecutorDifferentInputsOutputs<
                    NUM_LIMBS,
                    NUM_READ_OPS,
                    NUM_WRITE_OPS,
                    LIMB_BITS,
                >,
            >,
            (
                <BaseAluAdapterExecutorDifferentInputsOutputs<
                    NUM_LIMBS,
                    NUM_READ_OPS,
                    NUM_WRITE_OPS,
                    LIMB_BITS,
                > as AdapterTraceExecutor<F>>::RecordMut<'buf>,
                &'buf mut EqCoreRecord<NUM_LIMBS>,
            ),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", EqOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, openvm_circuit::system::memory::online::TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { opcode, .. } = instruction;

        let local_opcode = EqOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        BaseAluAdapterExecutorDifferentInputsOutputs::<
            NUM_LIMBS,
            NUM_READ_OPS,
            NUM_WRITE_OPS,
            LIMB_BITS,
        >::start(*state.pc, state.memory, &mut adapter_record);

        let [lhs, rhs] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let cmp_result = match local_opcode {
            EqOpcode::EQ => lhs == rhs,
            EqOpcode::NEQ => lhs != rhs,
        };

        let mut write_data = [[0u8; NUM_LIMBS]; 1];
        write_data[0][0] = cmp_result as u8;
        self.adapter.write(
            state.memory,
            instruction,
            write_data.into(),
            &mut adapter_record,
        );

        core_record.b = lhs;
        core_record.c = rhs;
        core_record.local_opcode = local_opcode as u8;

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

use openvm_circuit::arch::*;
use openvm_circuit::system::memory::online::{GuestMemory, TracingMemory};
use openvm_instructions::{LocalOpcode, instruction::Instruction};
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;
use std::borrow::{Borrow, BorrowMut};
use std::mem::size_of;

// Re-export upstream types that we don't modify
pub use openvm_rv32im_circuit::{
    BaseAluCoreAir, BaseAluCoreCols, BaseAluCoreRecord, BaseAluFiller,
};

// Our own BaseAluCoreExecutor that uses FP-aware adapters
#[derive(Clone, Copy, derive_new::new)]
pub struct BaseAluCoreExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub adapter: A,
    pub offset: usize,
}

// Helper function for ALU operations (not FP-related, pure ALU logic)
pub fn run_alu<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: BaseAluOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    use BaseAluOpcode::*;
    match opcode {
        ADD => {
            let mut result = [0u8; NUM_LIMBS];
            let mut carry = 0u16;
            for i in 0..NUM_LIMBS {
                let sum = x[i] as u16 + y[i] as u16 + carry;
                result[i] = sum as u8;
                carry = sum >> LIMB_BITS;
            }
            result
        }
        SUB => {
            let mut result = [0u8; NUM_LIMBS];
            let mut borrow = 0i16;
            for i in 0..NUM_LIMBS {
                let diff = x[i] as i16 - y[i] as i16 - borrow;
                result[i] = diff as u8;
                borrow = if diff < 0 { 1 } else { 0 };
            }
            result
        }
        XOR => std::array::from_fn(|i| x[i] ^ y[i]),
        OR => std::array::from_fn(|i| x[i] | y[i]),
        AND => std::array::from_fn(|i| x[i] & y[i]),
    }
}

use openvm_circuit_primitives::AlignedBytesBorrow;

// PreCompute struct for InterpreterExecutor
// This stores instruction parameters in a compact byte representation for zero-copy execution
#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct BaseAluPreCompute {
    local_opcode: u8,
    a: u8,
    b: u8,
    c: u32,
    e: u8,
}

// Execute function for InterpreterExecutor
// The unsafe code is required by OpenVM's InterpreterExecutor design - it uses raw pointers
// for zero-copy performance. This function is called via function pointer after pre_compute
// stores the instruction data as bytes.
unsafe fn execute_base_alu<
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, Ctx>,
) {
    use crate::adapters::{memory_read, memory_write};
    use openvm_instructions::{program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS};

    // Convert raw bytes back to BaseAluPreCompute struct (unsafe because raw pointer dereference)
    let pre_compute: &BaseAluPreCompute = unsafe {
        std::slice::from_raw_parts(pre_compute, std::mem::size_of::<BaseAluPreCompute>()).borrow()
    };

    let local_opcode = BaseAluOpcode::from_usize(pre_compute.local_opcode as usize);

    // Read operands
    let rs1_data: [u8; NUM_LIMBS] =
        memory_read(&exec_state.memory, RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2_data: [u8; NUM_LIMBS] = if pre_compute.e as u32 == RV32_REGISTER_AS {
        memory_read(&exec_state.memory, RV32_REGISTER_AS, pre_compute.c)
    } else {
        // Immediate value - sign extend to NUM_LIMBS bytes
        let imm_4bytes = pre_compute.c.to_le_bytes();
        let sign_byte = imm_4bytes[3];
        let mut result = [sign_byte; NUM_LIMBS];
        result[..4].copy_from_slice(&imm_4bytes);
        result
    };

    // Perform ALU operation
    let rd_data = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &rs1_data, &rs2_data);

    // Write result
    memory_write(
        &mut exec_state.memory,
        RV32_REGISTER_AS,
        pre_compute.a as u32,
        rd_data,
    );

    // Increment PC
    let next_pc = exec_state.pc().wrapping_add(DEFAULT_PC_STEP);
    exec_state.set_pc(next_pc);
}

// InterpreterExecutor implementation - required by OpenVM trait bounds but unused in FP-only system
impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for BaseAluCoreExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<BaseAluPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        use openvm_instructions::riscv::RV32_REGISTER_AS;

        let Instruction { a, b, c, d, e, .. } = *inst;
        let local_opcode = BaseAluOpcode::from_usize(inst.opcode.local_opcode_idx(self.offset));

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        let pre_compute = BaseAluPreCompute {
            local_opcode: local_opcode as u8,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32(),
            e: e.as_canonical_u32() as u8,
        };

        data[..std::mem::size_of::<BaseAluPreCompute>()].copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                &pre_compute as *const _ as *const u8,
                std::mem::size_of::<BaseAluPreCompute>(),
            )
        });

        Ok(execute_base_alu::<F, Ctx, NUM_LIMBS, LIMB_BITS>)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        use openvm_instructions::riscv::RV32_REGISTER_AS;

        let Instruction { a, b, c, d, e, .. } = *inst;
        let local_opcode = BaseAluOpcode::from_usize(inst.opcode.local_opcode_idx(self.offset));

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        let pre_compute = BaseAluPreCompute {
            local_opcode: local_opcode as u8,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32(),
            e: e.as_canonical_u32() as u8,
        };

        data[..std::mem::size_of::<BaseAluPreCompute>()].copy_from_slice(pre_compute.as_bytes());

        Ok(Box::new(execute_base_alu::<F, Ctx, NUM_LIMBS, LIMB_BITS>))
    }
}

// Metered execution function - similar to execute_base_alu but with chip_idx tracking
unsafe fn execute_base_alu_metered<
    F: PrimeField32,
    Ctx: MeteredExecutionCtxTrait,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, Ctx>,
) {
    use crate::adapters::{memory_read, memory_write};
    use openvm_instructions::{program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS};

    let pre_compute: &E2PreCompute<BaseAluPreCompute> = unsafe {
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<BaseAluPreCompute>>())
            .borrow()
    };

    // Track chip height
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);

    let data = &pre_compute.data;
    let local_opcode = BaseAluOpcode::from_usize(data.local_opcode as usize);

    // Read operands
    let rs1_data: [u8; NUM_LIMBS] =
        memory_read(&exec_state.memory, RV32_REGISTER_AS, data.b as u32);
    let rs2_data: [u8; NUM_LIMBS] = if data.e as u32 == RV32_REGISTER_AS {
        memory_read(&exec_state.memory, RV32_REGISTER_AS, data.c)
    } else {
        // Immediate value - sign extend to NUM_LIMBS bytes
        let imm_4bytes = data.c.to_le_bytes();
        let sign_byte = imm_4bytes[3];
        let mut result = [sign_byte; NUM_LIMBS];
        result[..4].copy_from_slice(&imm_4bytes);
        result
    };

    // Perform ALU operation
    let rd_data = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &rs1_data, &rs2_data);

    // Write result
    memory_write(
        &mut exec_state.memory,
        RV32_REGISTER_AS,
        data.a as u32,
        rd_data,
    );

    // Increment PC
    let next_pc = exec_state.pc().wrapping_add(DEFAULT_PC_STEP);
    exec_state.set_pc(next_pc);
}

// InterpreterMeteredExecutor implementation - required for proving
impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for BaseAluCoreExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BaseAluPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        use openvm_instructions::riscv::RV32_REGISTER_AS;

        let Instruction { a, b, c, d, e, .. } = *inst;
        let local_opcode = BaseAluOpcode::from_usize(inst.opcode.local_opcode_idx(self.offset));

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        let pre_compute: &mut E2PreCompute<BaseAluPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        pre_compute.data = BaseAluPreCompute {
            local_opcode: local_opcode as u8,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32(),
            e: e.as_canonical_u32() as u8,
        };

        Ok(execute_base_alu_metered::<F, Ctx, NUM_LIMBS, LIMB_BITS>)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        use openvm_instructions::riscv::RV32_REGISTER_AS;

        let Instruction { a, b, c, d, e, .. } = *inst;
        let local_opcode = BaseAluOpcode::from_usize(inst.opcode.local_opcode_idx(self.offset));

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        let pre_compute: &mut E2PreCompute<BaseAluPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        pre_compute.data = BaseAluPreCompute {
            local_opcode: local_opcode as u8,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32(),
            e: e.as_canonical_u32() as u8,
        };

        Ok(Box::new(
            execute_base_alu_metered::<F, Ctx, NUM_LIMBS, LIMB_BITS>,
        ))
    }
}

// FpPreflightExecutor implementation when adapter is FP-aware
impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> crate::FpPreflightExecutor<F, RA>
    for BaseAluCoreExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + crate::FpAdapterTraceExecutor<
            F,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
            'buf,
            EmptyAdapterCoreLayout<F, A>,
            (A::RecordMut<'buf>, &'buf mut BaseAluCoreRecord<NUM_LIMBS>),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", BaseAluOpcode::from_usize(opcode - self.offset))
    }

    fn execute_with_fp(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
        fp: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        use openvm_instructions::program::DEFAULT_PC_STEP;

        let Instruction { opcode, .. } = instruction;

        let local_opcode = BaseAluOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        // Call FP-aware start
        A::start_with_fp(*state.pc, fp, state.memory, &mut adapter_record);

        [core_record.b, core_record.c] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let rd = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &core_record.b, &core_record.c);

        core_record.local_opcode = local_opcode as u8;

        self.adapter
            .write(state.memory, instruction, [rd].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        // FP doesn't change for basic ALU operations
        Ok(None)
    }
}

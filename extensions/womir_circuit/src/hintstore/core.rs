use openvm_circuit::{arch::*, system::memory::online::TracingMemory};
use openvm_instructions::{LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::HintStoreOpcode;
use openvm_womir_transpiler::HintStoreOpcode::{HINT_BUFFER, HINT_STOREW};

use crate::adapters::{RV32_REGISTER_NUM_LIMBS, read_rv32_register, tracing_read, tracing_write};

// Import from upstream crate (re-exported at crate level)
use openvm_circuit::arch::{MultiRowLayout, MultiRowMetadata};
use openvm_rv32im_circuit::Rv32HintStoreRecordMut;

// Local metadata type (upstream's field is private, so we define our own)
#[derive(Copy, Clone, Debug)]
pub struct HintStoreMetadata {
    pub num_words: usize,
}

impl MultiRowMetadata for HintStoreMetadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        self.num_words
    }
}

pub type HintStoreLayout = MultiRowLayout<HintStoreMetadata>;

// Core executor that implements FpPreflightExecutor
#[derive(Clone, Copy, derive_new::new)]
pub struct HintStoreCoreExecutor {
    pub pointer_max_bits: usize,
    pub offset: usize,
}

// FpPreflightExecutor implementation for HintStore
impl<F, RA> crate::FpPreflightExecutor<F, RA> for HintStoreCoreExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, HintStoreLayout, Rv32HintStoreRecordMut<'buf>>,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        if opcode == HINT_STOREW.global_opcode().as_usize() {
            String::from("HINT_STOREW")
        } else if opcode == HINT_BUFFER.global_opcode().as_usize() {
            String::from("HINT_BUFFER")
        } else {
            unreachable!("unsupported opcode: {opcode}")
        }
    }

    fn execute_with_fp(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
        fp: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        let &Instruction {
            opcode, a, b, d, e, ..
        } = instruction;

        let a = a.as_canonical_u32();
        let b_base = b.as_canonical_u32();
        debug_assert_eq!(
            d.as_canonical_u32(),
            openvm_instructions::riscv::RV32_REGISTER_AS
        );
        debug_assert_eq!(
            e.as_canonical_u32(),
            openvm_instructions::riscv::RV32_MEMORY_AS
        );

        let local_opcode = HintStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        if local_opcode == HINT_STOREW {
            // HINT_STOREW: read from hint stream and write directly to register
            // hint_stream format: [length (4 bytes), data (4 bytes)]
            // Total: 8 bytes

            if state.streams.hint_stream.len() < RV32_REGISTER_NUM_LIMBS * 2 {
                return Err(ExecutionError::HintOutOfBounds { pc: *state.pc });
            }

            let record = state
                .ctx
                .alloc(HintStoreLayout::new(HintStoreMetadata { num_words: 1 }));

            record.inner.from_pc = *state.pc;
            record.inner.timestamp = state.memory.timestamp;
            record.inner.num_words = 1;
            record.inner.num_words_ptr = u32::MAX;

            // Read 8 bytes from hint stream: length (4) + data (4)
            let length_bytes: [F; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|_| state.streams.hint_stream.pop_front().unwrap());
            let data_f: [F; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|_| state.streams.hint_stream.pop_front().unwrap());

            // Verify length is 4
            debug_assert_eq!(length_bytes[0].as_canonical_u32(), 4);
            debug_assert_eq!(length_bytes[1].as_canonical_u32(), 0);
            debug_assert_eq!(length_bytes[2].as_canonical_u32(), 0);
            debug_assert_eq!(length_bytes[3].as_canonical_u32(), 0);

            let data: [u8; RV32_REGISTER_NUM_LIMBS] =
                data_f.map(|byte| byte.as_canonical_u32() as u8);

            record.var[0].data = data;

            // Write directly to register at address (a + fp) in register address space
            let target_reg = a + fp;
            record.inner.mem_ptr = target_reg;
            record.inner.mem_ptr_ptr = target_reg;

            state.memory.increment_timestamp();
            tracing_write(
                state.memory,
                openvm_instructions::riscv::RV32_REGISTER_AS, // Write to register space!
                target_reg,
                data,
                &mut record.var[0].data_write_aux.prev_timestamp,
                &mut record.var[0].data_write_aux.prev_data,
            );

            *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
            return Ok(None);
        }

        // HINT_BUFFER: read from registers to get memory pointer, write to memory
        let b = b_base + fp;

        let num_words = read_rv32_register(state.memory.data(), a + fp);

        let record = state.ctx.alloc(HintStoreLayout::new(HintStoreMetadata {
            num_words: num_words as usize,
        }));

        record.inner.from_pc = *state.pc;
        record.inner.timestamp = state.memory.timestamp;
        record.inner.mem_ptr_ptr = b;

        record.inner.mem_ptr = u32::from_le_bytes(tracing_read(
            state.memory,
            openvm_instructions::riscv::RV32_REGISTER_AS,
            b,
            &mut record.inner.mem_ptr_aux_record.prev_timestamp,
        ));

        debug_assert!(record.inner.mem_ptr <= (1 << self.pointer_max_bits));
        debug_assert_ne!(num_words, 0);
        debug_assert!(num_words <= (1 << self.pointer_max_bits));

        record.inner.num_words = num_words;
        record.inner.num_words_ptr = a + fp;
        tracing_read::<RV32_REGISTER_NUM_LIMBS>(
            state.memory,
            openvm_instructions::riscv::RV32_REGISTER_AS,
            record.inner.num_words_ptr,
            &mut record.inner.num_words_read.prev_timestamp,
        );

        if state.streams.hint_stream.len() < RV32_REGISTER_NUM_LIMBS * num_words as usize {
            return Err(ExecutionError::HintOutOfBounds { pc: *state.pc });
        }

        for idx in 0..(num_words as usize) {
            if idx != 0 {
                state.memory.increment_timestamp();
                state.memory.increment_timestamp();
            }

            let data_f: [F; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|_| state.streams.hint_stream.pop_front().unwrap());
            let data: [u8; RV32_REGISTER_NUM_LIMBS] =
                data_f.map(|byte| byte.as_canonical_u32() as u8);

            record.var[idx].data = data;

            tracing_write(
                state.memory,
                openvm_instructions::riscv::RV32_MEMORY_AS,
                record.inner.mem_ptr + (RV32_REGISTER_NUM_LIMBS * idx) as u32,
                data,
                &mut record.var[idx].data_write_aux.prev_timestamp,
                &mut record.var[idx].data_write_aux.prev_data,
            );
        }
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        // FP doesn't change for HintStore
        Ok(None)
    }
}

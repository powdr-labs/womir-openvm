use std::array;

use openvm_circuit::arch::*;
use openvm_circuit::system::memory::online::TracingMemory;
use openvm_instructions::{LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::PrimeField32;

// Re-export upstream types that we don't modify
pub use openvm_rv32im_circuit::{
    LoadStoreCoreAir, LoadStoreCoreCols, LoadStoreCoreRecord, LoadStoreFiller,
};

// Helper function for write data computation (pure logic, no FP dependency)
#[inline(always)]
pub(super) fn run_write_data<const NUM_CELLS: usize>(
    opcode: Rv32LoadStoreOpcode,
    read_data: [u8; NUM_CELLS],
    prev_data: [u32; NUM_CELLS],
    shift: usize,
) -> [u32; NUM_CELLS] {
    match (opcode, shift) {
        (LOADW, 0) => read_data.map(|x| x as u32),
        (LOADBU, 0) | (LOADBU, 1) | (LOADBU, 2) | (LOADBU, 3) => {
            let mut write_data = [0; NUM_CELLS];
            write_data[0] = read_data[shift] as u32;
            write_data
        }
        (LOADHU, 0) | (LOADHU, 2) => {
            let mut write_data = [0; NUM_CELLS];
            for (i, cell) in write_data.iter_mut().take(NUM_CELLS / 2).enumerate() {
                *cell = read_data[i + shift] as u32;
            }
            write_data
        }
        (STOREW, 0) => read_data.map(|x| x as u32),
        (STOREB, 0) | (STOREB, 1) | (STOREB, 2) | (STOREB, 3) => {
            let mut write_data = prev_data;
            write_data[shift] = read_data[0] as u32;
            write_data
        }
        (STOREH, 0) | (STOREH, 2) => array::from_fn(|i| {
            if i >= shift && i < (NUM_CELLS / 2 + shift) {
                read_data[i - shift] as u32
            } else {
                prev_data[i]
            }
        }),
        _ => unreachable!(
            "unaligned memory access not supported: {opcode:?}, shift: {shift}"
        ),
    }
}

// Our own LoadStoreExecutor that uses FP-aware adapters
#[derive(Clone, Copy, derive_new::new)]
pub struct LoadStoreExecutor<A, const NUM_CELLS: usize> {
    adapter: A,
    pub offset: usize,
}

// FpPreflightExecutor implementation when adapter is FP-aware
impl<F, A, RA, const NUM_CELLS: usize> crate::FpPreflightExecutor<F, RA> for LoadStoreExecutor<A, NUM_CELLS>
where
    F: PrimeField32,
    A: 'static
        + crate::FpAdapterTraceExecutor<
            F,
            ReadData = (([u32; NUM_CELLS], [u8; NUM_CELLS]), u8),
            WriteData = [u32; NUM_CELLS],
        >,
    for<'buf> RA: RecordArena<
            'buf,
            EmptyAdapterCoreLayout<F, A>,
            (A::RecordMut<'buf>, &'buf mut LoadStoreCoreRecord<NUM_CELLS>),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32LoadStoreOpcode::from_usize(opcode - self.offset)
        )
    }

    fn execute_with_fp(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
        fp: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        // Call FP-aware start
        A::start_with_fp(*state.pc, fp, state.memory, &mut adapter_record);

        (
            (core_record.prev_data, core_record.read_data),
            core_record.shift_amount,
        ) = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);

        let local_opcode = Rv32LoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        core_record.local_opcode = local_opcode as u8;

        let write_data = run_write_data(
            local_opcode,
            core_record.read_data,
            core_record.prev_data,
            core_record.shift_amount as usize,
        );
        self.adapter
            .write(state.memory, instruction, write_data, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        // FP doesn't change for loadstore operations
        Ok(None)
    }
}

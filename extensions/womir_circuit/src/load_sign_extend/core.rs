use openvm_circuit::arch::*;
use openvm_circuit::system::memory::online::TracingMemory;
use openvm_instructions::{LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, LOADB};
use openvm_stark_backend::p3_field::PrimeField32;

// Re-export upstream types that we don't modify
pub use openvm_rv32im_circuit::{
    LoadSignExtendCoreAir, LoadSignExtendCoreCols, LoadSignExtendCoreRecord, LoadSignExtendFiller,
};

// Helper function for sign-extended write data (pure logic, no FP dependency)
#[inline(always)]
pub(super) fn run_write_data_sign_extend<const NUM_CELLS: usize>(
    opcode: Rv32LoadStoreOpcode,
    read_data: [u8; NUM_CELLS],
    shift: usize,
) -> [u8; NUM_CELLS] {
    let mut write_data = [0u8; NUM_CELLS];
    if opcode == LOADB {
        // Load byte with sign extension
        let byte = read_data[shift];
        write_data[0] = byte;
        let sign_extend = if byte & 0x80 != 0 { 0xFF } else { 0x00 };
        for cell in write_data.iter_mut().skip(1) {
            *cell = sign_extend;
        }
    } else {
        // Load halfword with sign extension (shift must be 0 or 2)
        write_data[0] = read_data[shift];
        write_data[1] = read_data[shift + 1];
        let sign_extend = if read_data[shift + 1] & 0x80 != 0 {
            0xFF
        } else {
            0x00
        };
        for cell in write_data.iter_mut().skip(2) {
            *cell = sign_extend;
        }
    }
    write_data
}

// Our own LoadSignExtendExecutor (still uses AdapterTraceExecutor, not FP-aware)
#[derive(Clone, Copy, derive_new::new)]
pub struct LoadSignExtendExecutor<A, const NUM_CELLS: usize, const LIMB_BITS: usize> {
    adapter: A,
}

impl<F, A, RA, const NUM_CELLS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for LoadSignExtendExecutor<A, NUM_CELLS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData = (([u32; NUM_CELLS], [u8; NUM_CELLS]), u8),
            WriteData = [u32; NUM_CELLS],
        >,
    for<'buf> RA: RecordArena<
            'buf,
            EmptyAdapterCoreLayout<F, A>,
            (
                A::RecordMut<'buf>,
                &'buf mut LoadSignExtendCoreRecord<NUM_CELLS>,
            ),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32LoadStoreOpcode::from_usize(opcode - Rv32LoadStoreOpcode::CLASS_OFFSET)
        )
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let tmp = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);

        core_record.is_byte = local_opcode == LOADB;
        core_record.prev_data = tmp.0.0.map(|x| x as u8);
        core_record.read_data = tmp.0.1;
        core_record.shift_amount = tmp.1;

        let write_data = run_write_data_sign_extend(
            local_opcode,
            core_record.read_data,
            core_record.shift_amount as usize,
        );

        self.adapter.write(
            state.memory,
            instruction,
            write_data.map(u32::from),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

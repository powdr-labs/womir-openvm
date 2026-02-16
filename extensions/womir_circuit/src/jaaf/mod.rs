use std::borrow::BorrowMut;

use openvm_circuit::arch::{
    AdapterTraceFiller, TraceFiller, VmAirWrapper, VmChipWrapper, get_record_from_slice,
};
use openvm_instructions::LocalOpcode;
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_womir_transpiler::JaafOpcode;

use crate::adapters::jaaf::{JaafAdapterAir, JaafAdapterExecutor, JaafAdapterFiller};

mod core;
pub mod execution;

pub use self::core::{JaafCoreAir, JaafCoreCols, JaafCoreRecord};
pub use execution::JaafExecutor;

use openvm_circuit::system::memory::MemoryAuxColsFactory;

pub type Rv32JaafExecutor = JaafExecutor<JaafAdapterExecutor>;
pub type JaafAir = VmAirWrapper<JaafAdapterAir, JaafCoreAir>;
pub type JaafChip<F> = VmChipWrapper<F, JaafFiller>;

/// The filler for the JAAF chip, combining adapter filler and core filler.
#[derive(derive_new::new)]
pub struct JaafFiller {
    adapter: JaafAdapterFiller,
    pub offset: usize,
}

impl<F: PrimeField32> TraceFiller<F> for JaafFiller {
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // Split row into adapter and core parts
        let (adapter_row, mut core_row) = unsafe {
            row_slice.split_at_mut_unchecked(<JaafAdapterFiller as AdapterTraceFiller<F>>::WIDTH)
        };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        // Read the core record that was written by the preflight executor
        let record: &JaafCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut JaafCoreCols<F> = core_row.borrow_mut();

        // Fill in reverse order (to ensure non-overlapping with record)
        let local_opcode = JaafOpcode::from_usize(record.local_opcode as usize);

        // Opcode flags
        core_row.is_call_indirect = F::from_bool(local_opcode == JaafOpcode::CALL_INDIRECT);
        core_row.is_call = F::from_bool(local_opcode == JaafOpcode::CALL);
        core_row.is_ret = F::from_bool(local_opcode == JaafOpcode::RET);

        // return_pc_data
        core_row.return_pc_data = record.return_pc_data.map(F::from_canonical_u8);
        // old_fp_data
        core_row.old_fp_data = record.old_fp_data.map(F::from_canonical_u8);
        // to_pc_data
        core_row.to_pc_data = record.to_pc_data.map(F::from_canonical_u8);
        // new_fp_data
        core_row.new_fp_data = record.new_fp_data.map(F::from_canonical_u8);
    }
}

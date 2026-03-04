use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{Chip, prover::types::AirProvingContext};

use super::{Rv32HintStoreCols, Rv32HintStoreRecordHeader, Rv32HintStoreVar};
use crate::{adapters::RV32_CELL_BITS, cuda_abi::hintstore_cuda};

/// Reinterpret a `&[u32]` slice as `&[u8]` for GPU transfer.
fn as_u8_slice(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * size_of::<u32>()) }
}

#[derive(new)]
pub struct Rv32HintStoreChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32HintStoreChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }

        // Parse variable-size records to compute offsets and total rows.
        // Record layout: [Header (padded to Var alignment)] [Var * num_words]
        let header_size = size_of::<Rv32HintStoreRecordHeader>();
        let var_size = size_of::<Rv32HintStoreVar>();
        let aligned_header = header_size.next_multiple_of(align_of::<Rv32HintStoreVar>());

        let mut record_offsets: Vec<u32> = Vec::new();
        let mut row_offsets: Vec<u32> = Vec::new();
        let mut total_rows: u32 = 0;
        let mut offset = 0usize;

        while offset < records.len() {
            record_offsets.push(offset as u32);
            row_offsets.push(total_rows);

            // SAFETY: the executor wrote valid Rv32HintStoreRecordHeader at this offset
            let header: &Rv32HintStoreRecordHeader =
                unsafe { &*(records[offset..].as_ptr() as *const Rv32HintStoreRecordHeader) };
            let num_words = header.num_words;
            debug_assert!(num_words > 0);
            total_rows += num_words;

            offset += aligned_header + var_size * num_words as usize;
        }
        debug_assert_eq!(offset, records.len());

        let trace_width = Rv32HintStoreCols::<F>::width();
        let padded_height = next_power_of_two_or_zero(total_rows as usize);

        // Copy data to GPU
        let d_records = records.to_device().unwrap();
        let d_record_offsets = as_u8_slice(&record_offsets).to_device().unwrap();
        let d_row_offsets = as_u8_slice(&row_offsets).to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(padded_height, trace_width);

        unsafe {
            hintstore_cuda::tracegen(
                d_trace.buffer(),
                padded_height,
                trace_width,
                &d_records,
                &d_record_offsets,
                &d_row_offsets,
                record_offsets.len() as u32,
                total_rows,
                self.pointer_max_bits as u32,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

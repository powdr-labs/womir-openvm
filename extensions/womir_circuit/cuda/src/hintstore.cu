// CUDA tracegen for HintStore chip (multi-row).
// One thread per instruction; each thread fills num_words consecutive rows.
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "womir/hintstore.cuh"

// Combined kernel: threads [0, num_instructions) fill trace rows from records,
// threads [num_instructions, height) zero-fill padding rows.
__global__ void womir_hintstore_tracegen(
    Fp *d_trace,
    size_t height,
    uint8_t const *d_records,
    uint32_t const *d_record_offsets,
    uint32_t const *d_row_offsets,
    uint32_t num_instructions,
    uint32_t total_rows,
    uint32_t pointer_max_bits,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_instructions) {
        // Fill trace rows for instruction idx
        uint32_t rec_offset = d_record_offsets[idx];
        uint32_t row_offset = d_row_offsets[idx];

        auto const &header = *reinterpret_cast<HintStoreRecordHeader const *>(d_records + rec_offset);

        // Vars start after header, aligned to alignof(HintStoreVar)
        constexpr size_t header_size = sizeof(HintStoreRecordHeader);
        constexpr size_t var_align = alignof(HintStoreVar);
        constexpr size_t aligned_header = (header_size + var_align - 1) & ~(var_align - 1);
        auto const *vars = reinterpret_cast<HintStoreVar const *>(d_records + rec_offset + aligned_header);

        HintStoreTraceFiller filler(
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits,
            pointer_max_bits
        );
        filler.fill_rows(d_trace + row_offset, height, header, vars);
    }

    // Zero-fill padding rows: remap thread index to padding range
    // We use threads [0, height - total_rows) for this purpose.
    // Since kernel is launched with max(num_instructions, height - total_rows) threads,
    // all threads check both ranges.
    uint32_t pad_idx = idx;
    uint32_t num_padding = height - total_rows;
    if (pad_idx < num_padding) {
        RowSlice row(d_trace + (total_rows + pad_idx), height);
        row.fill_zero(0, sizeof(HintStoreCols<uint8_t>));
    }
}

extern "C" int _womir_hintstore_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t const *d_records,
    uint32_t const *d_record_offsets,
    uint32_t const *d_row_offsets,
    uint32_t num_instructions,
    uint32_t total_rows,
    uint32_t pointer_max_bits,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= total_rows);
    assert(width == sizeof(HintStoreCols<uint8_t>));

    // Launch enough threads for both instruction processing and padding
    uint32_t num_padding = height - total_rows;
    uint32_t num_threads = num_instructions > num_padding ? num_instructions : num_padding;
    if (num_threads == 0) {
        return 0;
    }
    auto [grid, block] = kernel_launch_params(num_threads);
    womir_hintstore_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_record_offsets,
        d_row_offsets,
        num_instructions,
        total_rows,
        pointer_max_bits,
        d_range_checker_ptr,
        range_checker_num_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

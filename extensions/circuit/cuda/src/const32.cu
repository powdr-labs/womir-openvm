#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "womir/const32.cuh"

__global__ void womir_const32_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Const32Record> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Const32TraceFiller filler(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        filler.fill_trace_row(row, rec);
    } else {
        row.fill_zero(0, sizeof(Const32Cols<uint8_t>));
    }
}

extern "C" int _womir_const32_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Const32Record> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(Const32Cols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    womir_const32_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

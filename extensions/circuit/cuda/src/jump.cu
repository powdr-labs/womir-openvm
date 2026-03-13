#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "womir/constants.cuh"
#include "womir/adapters/jump.cuh"
#include "womir/cores/jump.cuh"

template <typename T> struct WomirJumpCols {
    WomirJumpAdapterCols<T> adapter;
    JumpCoreCols<T> core;
};

struct WomirJumpRecord {
    WomirJumpAdapterRecord adapter;
    JumpCoreRecord core;
};

__global__ void womir_jump_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<WomirJumpRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        WomirJumpAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        JumpCore core;
        core.fill_trace_row(row.slice_from(COL_INDEX(WomirJumpCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(WomirJumpCols<uint8_t>));
    }
}

extern "C" int _womir_jump_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<WomirJumpRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(WomirJumpCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    womir_jump_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

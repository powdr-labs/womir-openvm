#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "crush/constants.cuh"
#include "crush/adapters/jump.cuh"
#include "crush/cores/jump.cuh"

template <typename T> struct CrushJumpCols {
    CrushJumpAdapterCols<T> adapter;
    JumpCoreCols<T> core;
};

struct CrushJumpRecord {
    CrushJumpAdapterRecord adapter;
    JumpCoreRecord core;
};

__global__ void crush_jump_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<CrushJumpRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        CrushJumpAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        JumpCore core;
        core.fill_trace_row(row.slice_from(COL_INDEX(CrushJumpCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(CrushJumpCols<uint8_t>));
    }
}

extern "C" int _crush_jump_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<CrushJumpRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(CrushJumpCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    crush_jump_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

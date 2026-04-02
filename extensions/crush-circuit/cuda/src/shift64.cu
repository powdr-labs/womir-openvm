#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "crush/constants.cuh"
#include "crush/adapters/alu.cuh"
#include "rv32im/cores/shift.cuh"

// Concrete type aliases for 64-bit
using CrushShift64CoreRecord = ShiftCoreRecord<W64_NUM_LIMBS>;
using CrushShift64Core = ShiftCore<W64_NUM_LIMBS>;
template <typename T> using CrushShift64CoreCols = ShiftCoreCols<T, W64_NUM_LIMBS>;

template <typename T> struct CrushShift64Cols {
    CrushBaseAluAdapterCols<T, W64_REG_OPS, W64_REG_OPS> adapter;
    CrushShift64CoreCols<T> core;
};

struct CrushShift64Record {
    CrushBaseAluAdapterRecord<W64_REG_OPS, W64_REG_OPS> adapter;
    CrushShift64CoreRecord core;
};

__global__ void crush_shift64_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<CrushShift64Record> d_records,
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

        CrushBaseAluAdapter<W64_REG_OPS, W64_REG_OPS> adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        CrushShift64Core core(
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(CrushShift64Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(CrushShift64Cols<uint8_t>));
    }
}

extern "C" int _crush_shift64_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<CrushShift64Record> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(CrushShift64Cols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    crush_shift64_tracegen<<<grid, block>>>(
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

#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "crush/constants.cuh"
#include "crush/adapters/alu.cuh"
#include "crush/cores/alu.cuh"

// Concrete type aliases for 64-bit
using CrushBaseAlu64CoreRecord = BaseAluCoreRecord<W64_NUM_LIMBS>;
using CrushBaseAlu64Core = BaseAluCore<W64_NUM_LIMBS>;
template <typename T> using CrushBaseAlu64CoreCols = BaseAluCoreCols<T, W64_NUM_LIMBS>;

template <typename T> struct CrushBaseAlu64Cols {
    CrushBaseAluAdapterCols<T, W64_REG_OPS, W64_REG_OPS> adapter;
    CrushBaseAlu64CoreCols<T> core;
};

struct CrushBaseAlu64Record {
    CrushBaseAluAdapterRecord<W64_REG_OPS, W64_REG_OPS> adapter;
    CrushBaseAlu64CoreRecord core;
};

__global__ void crush_alu64_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<CrushBaseAlu64Record> d_records,
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

        CrushBaseAlu64Core core(BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row(row.slice_from(COL_INDEX(CrushBaseAlu64Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(CrushBaseAlu64Cols<uint8_t>));
    }
}

extern "C" int _crush_alu64_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<CrushBaseAlu64Record> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(CrushBaseAlu64Cols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    crush_alu64_tracegen<<<grid, block>>>(
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

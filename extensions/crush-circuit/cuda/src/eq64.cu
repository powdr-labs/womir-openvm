#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "crush/constants.cuh"
#include "crush/adapters/alu.cuh"
#include "crush/cores/eq.cuh"

// Concrete type aliases for 64-bit
using CrushEq64CoreRecord = EqCoreRecord<W64_NUM_LIMBS>;
using CrushEq64Core = EqCore<W64_NUM_LIMBS>;
template <typename T> using CrushEq64CoreCols = EqCoreCols<T, W64_NUM_LIMBS>;

// 64-bit EQ reads two 64-bit operands (W64_REG_OPS=2 reads per operand)
// but writes only one 32-bit result (W32_REG_OPS=1 write).
template <typename T> struct CrushEq64Cols {
    CrushBaseAluAdapterCols<T, W64_REG_OPS, W32_REG_OPS> adapter;
    CrushEq64CoreCols<T> core;
};

struct CrushEq64Record {
    CrushBaseAluAdapterRecord<W64_REG_OPS, W32_REG_OPS> adapter;
    CrushEq64CoreRecord core;
};

__global__ void crush_eq64_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<CrushEq64Record> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = CrushBaseAluAdapter<W64_REG_OPS, W32_REG_OPS>(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        CrushEq64Core core;
        core.fill_trace_row(row.slice_from(COL_INDEX(CrushEq64Cols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(CrushEq64Cols<uint8_t>));
    }
}

extern "C" int _crush_eq64_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<CrushEq64Record> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(CrushEq64Cols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    crush_eq64_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

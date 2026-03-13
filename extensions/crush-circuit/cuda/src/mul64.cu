#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "crush/constants.cuh"
#include "crush/adapters/alu.cuh"
#include "rv32im/cores/mul.cuh"

// Concrete type aliases for 64-bit
using CrushMul64CoreRecord = MultiplicationCoreRecord<W64_NUM_LIMBS>;
using CrushMul64Core = MultiplicationCore<W64_NUM_LIMBS>;
template <typename T> using CrushMul64CoreCols = MultiplicationCoreCols<T, W64_NUM_LIMBS>;

template <typename T> struct CrushMul64Cols {
    CrushBaseAluAdapterCols<T, W64_REG_OPS, W64_REG_OPS> adapter;
    CrushMul64CoreCols<T> core;
};

struct CrushMul64Record {
    CrushBaseAluAdapterRecord<W64_REG_OPS, W64_REG_OPS> adapter;
    CrushMul64CoreRecord core;
};

__global__ void crush_mul64_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<CrushMul64Record> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
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

        RangeTupleChecker<2> range_tuple_checker(
            d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        CrushMul64Core core(range_tuple_checker);
        core.fill_trace_row(row.slice_from(COL_INDEX(CrushMul64Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(CrushMul64Cols<uint8_t>));
    }
}

extern "C" int _crush_mul64_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<CrushMul64Record> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(CrushMul64Cols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    crush_mul64_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        d_range_tuple_ptr,
        range_tuple_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

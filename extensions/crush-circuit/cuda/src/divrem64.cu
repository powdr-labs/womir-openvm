// 64-bit variant of divrem.cu.
// Uses CrushBaseAluAdapter with W64_REG_OPS and DivRemCore with W64_NUM_LIMBS.
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "crush/constants.cuh"
#include "crush/adapters/alu.cuh"
#include "rv32im/cores/divrem.cuh"

// Concrete type aliases for 64-bit
using CrushDivRem64CoreRecord = DivRemCoreRecords<W64_NUM_LIMBS>;
using CrushDivRem64Core = DivRemCore<W64_NUM_LIMBS>;
template <typename T> using CrushDivRem64CoreCols = DivRemCoreCols<T, W64_NUM_LIMBS>;

template <typename T> struct CrushDivRem64Cols {
    CrushBaseAluAdapterCols<T, W64_REG_OPS, W64_REG_OPS> adapter;
    CrushDivRem64CoreCols<T> core;
};

struct CrushDivRem64Record {
    CrushBaseAluAdapterRecord<W64_REG_OPS, W64_REG_OPS> adapter;
    CrushDivRem64CoreRecord core;
};

__global__ void crush_divrem64_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<CrushDivRem64Record> d_records,
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

        CrushDivRem64Core core(
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            RangeTupleChecker<2>(
                d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
            )
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(CrushDivRem64Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(CrushDivRem64Cols<uint8_t>));
    }
}

extern "C" int _crush_divrem64_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<CrushDivRem64Record> d_records,
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
    assert(width == sizeof(CrushDivRem64Cols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    crush_divrem64_tracegen<<<grid, block>>>(
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

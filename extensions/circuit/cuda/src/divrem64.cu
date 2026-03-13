// 64-bit variant of divrem.cu.
// Uses WomirBaseAluAdapter with W64_REG_OPS and DivRemCore with W64_NUM_LIMBS.
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "womir/constants.cuh"
#include "womir/adapters/alu.cuh"
#include "rv32im/cores/divrem.cuh"

// Concrete type aliases for 64-bit
using WomirDivRem64CoreRecord = DivRemCoreRecords<W64_NUM_LIMBS>;
using WomirDivRem64Core = DivRemCore<W64_NUM_LIMBS>;
template <typename T> using WomirDivRem64CoreCols = DivRemCoreCols<T, W64_NUM_LIMBS>;

template <typename T> struct WomirDivRem64Cols {
    WomirBaseAluAdapterCols<T, W64_REG_OPS, W64_REG_OPS> adapter;
    WomirDivRem64CoreCols<T> core;
};

struct WomirDivRem64Record {
    WomirBaseAluAdapterRecord<W64_REG_OPS, W64_REG_OPS> adapter;
    WomirDivRem64CoreRecord core;
};

__global__ void womir_divrem64_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<WomirDivRem64Record> d_records,
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

        WomirBaseAluAdapter<W64_REG_OPS, W64_REG_OPS> adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        WomirDivRem64Core core(
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            RangeTupleChecker<2>(
                d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
            )
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(WomirDivRem64Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(WomirDivRem64Cols<uint8_t>));
    }
}

extern "C" int _womir_divrem64_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<WomirDivRem64Record> d_records,
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
    assert(width == sizeof(WomirDivRem64Cols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    womir_divrem64_tracegen<<<grid, block>>>(
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

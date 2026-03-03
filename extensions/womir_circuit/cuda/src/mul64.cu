#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "womir/adapters/alu.cuh"
#include "rv32im/cores/mul.cuh"

using namespace riscv;

// 64-bit constants
static const size_t W64_NUM_LIMBS = 2 * RV32_REGISTER_NUM_LIMBS;
static const size_t W64_REG_OPS = 2;

// Concrete type aliases for 64-bit
using WomirMul64CoreRecord = MultiplicationCoreRecord<W64_NUM_LIMBS>;
using WomirMul64Core = MultiplicationCore<W64_NUM_LIMBS>;
template <typename T> using WomirMul64CoreCols = MultiplicationCoreCols<T, W64_NUM_LIMBS>;

template <typename T> struct WomirMul64Cols {
    WomirBaseAluAdapterCols<T, W64_REG_OPS, W64_REG_OPS> adapter;
    WomirMul64CoreCols<T> core;
};

struct WomirMul64Record {
    WomirBaseAluAdapterRecord<W64_REG_OPS, W64_REG_OPS> adapter;
    WomirMul64CoreRecord core;
};

__global__ void womir_mul64_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<WomirMul64Record> d_records,
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

        RangeTupleChecker<2> range_tuple_checker(
            d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        WomirMul64Core core(range_tuple_checker);
        core.fill_trace_row(row.slice_from(COL_INDEX(WomirMul64Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(WomirMul64Cols<uint8_t>));
    }
}

extern "C" int _womir_mul64_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<WomirMul64Record> d_records,
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
    assert(width == sizeof(WomirMul64Cols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    womir_mul64_tracegen<<<grid, block>>>(
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

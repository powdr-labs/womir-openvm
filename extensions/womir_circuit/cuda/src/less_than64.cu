#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "womir/constants.cuh"
#include "womir/adapters/alu.cuh"
#include "rv32im/cores/less_than.cuh"

// Concrete type aliases for 64-bit
using WomirLessThan64CoreRecord = LessThanCoreRecord<W64_NUM_LIMBS>;
using WomirLessThan64Core = LessThanCore<W64_NUM_LIMBS>;
template <typename T> using WomirLessThan64CoreCols = LessThanCoreCols<T, W64_NUM_LIMBS>;

// 64-bit LT reads two 64-bit operands (W64_REG_OPS=2 reads per operand)
// but writes only one 32-bit result (W32_REG_OPS=1 write).
template <typename T> struct WomirLessThan64Cols {
    WomirBaseAluAdapterCols<T, W64_REG_OPS, W32_REG_OPS> adapter;
    WomirLessThan64CoreCols<T> core;
};

struct WomirLessThan64Record {
    WomirBaseAluAdapterRecord<W64_REG_OPS, W32_REG_OPS> adapter;
    WomirLessThan64CoreRecord core;
};

__global__ void womir_less_than64_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<WomirLessThan64Record> d_records,
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

        WomirBaseAluAdapter<W64_REG_OPS, W32_REG_OPS> adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        WomirLessThan64Core core(BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row(row.slice_from(COL_INDEX(WomirLessThan64Cols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(WomirLessThan64Cols<uint8_t>));
    }
}

extern "C" int _womir_less_than64_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<WomirLessThan64Record> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(WomirLessThan64Cols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    womir_less_than64_tracegen<<<grid, block>>>(
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

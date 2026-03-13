#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "womir/constants.cuh"
#include "womir/adapters/alu.cuh"
#include "womir/cores/eq.cuh"

// Concrete type aliases for 64-bit
using WomirEq64CoreRecord = EqCoreRecord<W64_NUM_LIMBS>;
using WomirEq64Core = EqCore<W64_NUM_LIMBS>;
template <typename T> using WomirEq64CoreCols = EqCoreCols<T, W64_NUM_LIMBS>;

// 64-bit EQ reads two 64-bit operands (W64_REG_OPS=2 reads per operand)
// but writes only one 32-bit result (W32_REG_OPS=1 write).
template <typename T> struct WomirEq64Cols {
    WomirBaseAluAdapterCols<T, W64_REG_OPS, W32_REG_OPS> adapter;
    WomirEq64CoreCols<T> core;
};

struct WomirEq64Record {
    WomirBaseAluAdapterRecord<W64_REG_OPS, W32_REG_OPS> adapter;
    WomirEq64CoreRecord core;
};

__global__ void womir_eq64_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<WomirEq64Record> records,
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

        auto adapter = WomirBaseAluAdapter<W64_REG_OPS, W32_REG_OPS>(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        WomirEq64Core core;
        core.fill_trace_row(row.slice_from(COL_INDEX(WomirEq64Cols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(WomirEq64Cols<uint8_t>));
    }
}

extern "C" int _womir_eq64_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<WomirEq64Record> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(WomirEq64Cols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    womir_eq64_tracegen<<<grid, block>>>(
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

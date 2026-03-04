#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "womir/constants.cuh"
#include "womir/adapters/alu.cuh"
#include "rv32im/cores/less_than.cuh"

// Concrete type aliases for 32-bit
using WomirLessThanCoreRecord = LessThanCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using WomirLessThanCore = LessThanCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T> using WomirLessThanCoreCols = LessThanCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct WomirLessThanCols {
    WomirBaseAluAdapterCols<T, W32_REG_OPS, W32_REG_OPS> adapter;
    WomirLessThanCoreCols<T> core;
};

struct WomirLessThanRecord {
    WomirBaseAluAdapterRecord<W32_REG_OPS, W32_REG_OPS> adapter;
    WomirLessThanCoreRecord core;
};

__global__ void womir_less_than_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<WomirLessThanRecord> records,
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

        auto adapter = WomirBaseAluAdapter<W32_REG_OPS, W32_REG_OPS>(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = WomirLessThanCore(BitwiseOperationLookup(bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row(row.slice_from(COL_INDEX(WomirLessThanCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(WomirLessThanCols<uint8_t>));
    }
}

extern "C" int _womir_less_than_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<WomirLessThanRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(WomirLessThanCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    womir_less_than_tracegen<<<grid, block>>>(
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

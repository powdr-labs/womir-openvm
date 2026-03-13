#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "crush/constants.cuh"
#include "crush/adapters/alu.cuh"
#include "rv32im/cores/shift.cuh"

// Concrete type aliases for 32-bit
using CrushShiftCoreRecord = ShiftCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using CrushShiftCore = ShiftCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T> using CrushShiftCoreCols = ShiftCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct CrushShiftCols {
    CrushBaseAluAdapterCols<T, W32_REG_OPS, W32_REG_OPS> adapter;
    CrushShiftCoreCols<T> core;
};

struct CrushShiftRecord {
    CrushBaseAluAdapterRecord<W32_REG_OPS, W32_REG_OPS> adapter;
    CrushShiftCoreRecord core;
};

__global__ void crush_shift_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<CrushShiftRecord> d_records,
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

        CrushBaseAluAdapter<W32_REG_OPS, W32_REG_OPS> adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        CrushShiftCore core(
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(CrushShiftCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(CrushShiftCols<uint8_t>));
    }
}

extern "C" int _crush_shift_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<CrushShiftRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(CrushShiftCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    crush_shift_tracegen<<<grid, block>>>(
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

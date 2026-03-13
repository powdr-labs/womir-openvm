// Adapted from <openvm>/extensions/rv32im/circuit/cuda/src/alu.cu (namespace renames only)
// Diff: https://gist.github.com/leonardoalt/09fd3d60bd571851bb656dc53cec0a4b#file-diff-alu-cu-diff
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "crush/constants.cuh"
#include "crush/adapters/alu.cuh"
#include "crush/cores/alu.cuh"

// Concrete type aliases for 32-bit
using CrushBaseAluCoreRecord = BaseAluCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using CrushBaseAluCore = BaseAluCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T> using CrushBaseAluCoreCols = BaseAluCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct CrushBaseAluCols {
    CrushBaseAluAdapterCols<T, W32_REG_OPS, W32_REG_OPS> adapter;
    CrushBaseAluCoreCols<T> core;
};

struct CrushBaseAluRecord {
    CrushBaseAluAdapterRecord<W32_REG_OPS, W32_REG_OPS> adapter;
    CrushBaseAluCoreRecord core;
};

__global__ void crush_alu_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<CrushBaseAluRecord> d_records,
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

        CrushBaseAluCore core(BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row(row.slice_from(COL_INDEX(CrushBaseAluCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(CrushBaseAluCols<uint8_t>));
    }
}

extern "C" int _crush_alu_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<CrushBaseAluRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(CrushBaseAluCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    crush_alu_tracegen<<<grid, block>>>(
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

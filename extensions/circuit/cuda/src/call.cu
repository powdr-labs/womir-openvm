// GPU tracegen for the Call chip (CALL, CALL_INDIRECT, RET).
// Uses WomirCallAdapter (with frame pointer) and inlined CallCore logic.
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "womir/adapters/call.cuh"

using namespace riscv;

// CallOpcode enum - mirrors Rust CallOpcode
enum CallOpcode {
    RET = 0,
    CALL = 1,
    CALL_INDIRECT = 2,
};

// Core columns - mirrors CallCoreCols<T> in Rust
template <typename T> struct CallCoreCols {
    T new_fp_data[RV32_REGISTER_NUM_LIMBS];
    T to_pc_data[RV32_REGISTER_NUM_LIMBS];
    T old_fp_data[RV32_REGISTER_NUM_LIMBS];
    T return_pc_data[RV32_REGISTER_NUM_LIMBS];
    T is_ret;
    T is_call;
    T is_call_indirect;
};

// Core record - mirrors CallCoreRecord in Rust
struct CallCoreRecord {
    uint8_t new_fp_data[RV32_REGISTER_NUM_LIMBS];
    uint8_t to_pc_data[RV32_REGISTER_NUM_LIMBS];
    uint8_t old_fp_data[RV32_REGISTER_NUM_LIMBS];
    uint8_t return_pc_data[RV32_REGISTER_NUM_LIMBS];
    uint8_t local_opcode;
};

struct CallCore {
    template <typename T> using Cols = CallCoreCols<T>;

    __device__ void fill_trace_row(RowSlice row, CallCoreRecord record) {
        CallOpcode opcode = static_cast<CallOpcode>(record.local_opcode);

        COL_WRITE_ARRAY(row, Cols, new_fp_data, record.new_fp_data);
        COL_WRITE_ARRAY(row, Cols, to_pc_data, record.to_pc_data);
        COL_WRITE_ARRAY(row, Cols, old_fp_data, record.old_fp_data);
        COL_WRITE_ARRAY(row, Cols, return_pc_data, record.return_pc_data);

        COL_WRITE_VALUE(row, Cols, is_ret, opcode == RET);
        COL_WRITE_VALUE(row, Cols, is_call, opcode == CALL);
        COL_WRITE_VALUE(row, Cols, is_call_indirect, opcode == CALL_INDIRECT);
    }
};

// Combined columns and record
template <typename T> struct WomirCallCols {
    WomirCallAdapterCols<T> adapter;
    CallCoreCols<T> core;
};

struct WomirCallRecord {
    WomirCallAdapterRecord adapter;
    CallCoreRecord core;
};

__global__ void womir_call_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<WomirCallRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = WomirCallAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = CallCore();
        core.fill_trace_row(row.slice_from(COL_INDEX(WomirCallCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(WomirCallCols<uint8_t>));
    }
}

extern "C" int _womir_call_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<WomirCallRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(WomirCallCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);

    womir_call_tracegen<<<grid, block>>>(
        d_trace,
        height,
        width,
        d_records,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

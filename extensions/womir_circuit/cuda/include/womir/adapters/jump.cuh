// WOMIR Jump adapter for CUDA tracegen.
// Reads FP + 1 register (condition/offset), no writes.
// Simpler than the ALU adapter: single register read, no rs2, no write.
#pragma once

#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"
#include "womir/execution.cuh"

using namespace riscv;

template <typename T> struct WomirJumpAdapterCols {
    WomirExecutionState<T> from_state;
    T rs_ptr;
    MemoryReadAuxCols<T> fp_read_aux;
    MemoryReadAuxCols<T> rs_read_aux;
};

struct WomirJumpAdapterRecord {
    uint32_t from_pc;
    uint32_t fp;
    uint32_t from_timestamp;
    uint32_t rs_ptr;
    MemoryReadAuxRecord fp_read_aux;
    MemoryReadAuxRecord rs_read_aux;
};

struct WomirJumpAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ WomirJumpAdapter(
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, WomirJumpAdapterRecord record) {
        // rs read (at from_timestamp + 1, after fp read)
        mem_helper.fill(
            row.slice_from(COL_INDEX(WomirJumpAdapterCols, rs_read_aux)),
            record.rs_read_aux.prev_timestamp,
            record.from_timestamp + 1
        );

        // fp read (at from_timestamp + 0)
        mem_helper.fill(
            row.slice_from(COL_INDEX(WomirJumpAdapterCols, fp_read_aux)),
            record.fp_read_aux.prev_timestamp,
            record.from_timestamp
        );

        COL_WRITE_VALUE(row, WomirJumpAdapterCols, rs_ptr, record.rs_ptr);
        COL_WRITE_VALUE(row, WomirJumpAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, WomirJumpAdapterCols, from_state.fp, record.fp);
        COL_WRITE_VALUE(row, WomirJumpAdapterCols, from_state.pc, record.from_pc);
    }
};

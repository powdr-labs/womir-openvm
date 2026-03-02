#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

// WOMIR ExecutionState includes frame pointer (fp) between pc and timestamp.
template <typename T> struct WomirExecutionState {
    T pc;
    T fp;
    T timestamp;
};

template <typename T> struct WomirBaseAluAdapterCols {
    WomirExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    T rs2;    // Pointer if rs2 was a read, immediate value otherwise
    T rs2_as; // 1 if rs2 was a read, 0 if an immediate
    MemoryReadAuxCols<T> fp_read_aux;
    MemoryReadAuxCols<T> reads_aux[2];
    MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS> writes_aux;
};

struct WomirBaseAluAdapterRecord {
    uint32_t from_pc;
    uint32_t fp;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2;   // Pointer if rs2 was a read, immediate value otherwise
    uint8_t rs2_as; // 1 if rs2 was a read, 0 if an immediate
    MemoryReadAuxRecord fp_read_aux;
    MemoryReadAuxRecord reads_aux[2];
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> writes_aux;
};

struct WomirBaseAluAdapter {
    MemoryAuxColsFactory mem_helper;
    BitwiseOperationLookup bitwise_lookup;

    __device__ WomirBaseAluAdapter(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup lookup,
        uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits), bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, WomirBaseAluAdapterRecord record) {
        COL_WRITE_VALUE(row, WomirBaseAluAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, WomirBaseAluAdapterCols, from_state.fp, record.fp);
        COL_WRITE_VALUE(row, WomirBaseAluAdapterCols, from_state.timestamp, record.from_timestamp);

        COL_WRITE_VALUE(row, WomirBaseAluAdapterCols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(row, WomirBaseAluAdapterCols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, WomirBaseAluAdapterCols, rs2, record.rs2);
        COL_WRITE_VALUE(row, WomirBaseAluAdapterCols, rs2_as, record.rs2_as);

        // Read auxiliary for fp (at from_timestamp + 0)
        mem_helper.fill(
            row.slice_from(COL_INDEX(WomirBaseAluAdapterCols, fp_read_aux)),
            record.fp_read_aux.prev_timestamp,
            record.from_timestamp
        );

        // Read auxiliary for rs1 (at from_timestamp + 1)
        mem_helper.fill(
            row.slice_from(COL_INDEX(WomirBaseAluAdapterCols, reads_aux[0])),
            record.reads_aux[0].prev_timestamp,
            record.from_timestamp + 1
        );

        // rs2: register read when rs2_as == RV32_REGISTER_AS (== 1), otherwise immediate.
        if (record.rs2_as != 0) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(WomirBaseAluAdapterCols, reads_aux[1])),
                record.reads_aux[1].prev_timestamp,
                record.from_timestamp + 2
            );
        } else {
            RowSlice rs2_aux = row.slice_from(COL_INDEX(WomirBaseAluAdapterCols, reads_aux[1]));
#pragma unroll
            for (size_t i = 0; i < sizeof(MemoryReadAuxCols<uint8_t>); i++) {
                rs2_aux.write(i, 0);
            }
            uint32_t mask = (1u << RV32_CELL_BITS) - 1u;
            bitwise_lookup.add_range(record.rs2 & mask, (record.rs2 >> RV32_CELL_BITS) & mask);
        }

        COL_WRITE_ARRAY(
            row, WomirBaseAluAdapterCols, writes_aux.prev_data, record.writes_aux.prev_data
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(WomirBaseAluAdapterCols, writes_aux)),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 3
        );
    }
};

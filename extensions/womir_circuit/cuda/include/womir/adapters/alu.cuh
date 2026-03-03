// Adapted from <openvm>/extensions/rv32im/circuit/cuda/include/rv32im/adapters/alu.cuh
// Main changes: adds frame pointer (fp) field, WomirExecutionState, fp_read_aux, timestamp +1 shift
// Diff: https://gist.github.com/leonardoalt/09fd3d60bd571851bb656dc53cec0a4b#file-diff-adapters-alu-cuh-diff
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

template <typename T, size_t NUM_READ_OPS, size_t NUM_WRITE_OPS>
struct WomirBaseAluAdapterCols {
    WomirExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    T rs2;    // Pointer if rs2 was a read, immediate value otherwise
    T rs2_as; // 1 if rs2 was a read, 0 if an immediate
    MemoryReadAuxCols<T> fp_read_aux;
    MemoryReadAuxCols<T> rs1_reads_aux[NUM_READ_OPS];
    MemoryReadAuxCols<T> rs2_reads_aux[NUM_READ_OPS];
    MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS> writes_aux[NUM_WRITE_OPS];
};

template <size_t NUM_READ_OPS, size_t NUM_WRITE_OPS>
struct WomirBaseAluAdapterRecord {
    uint32_t from_pc;
    uint32_t fp;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2;   // Pointer if rs2 was a read, immediate value otherwise
    uint8_t rs2_as; // 1 if rs2 was a read, 0 if an immediate
    MemoryReadAuxRecord fp_read_aux;
    MemoryReadAuxRecord rs1_reads_aux[NUM_READ_OPS];
    MemoryReadAuxRecord rs2_reads_aux[NUM_READ_OPS];
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> writes_aux[NUM_WRITE_OPS];
};

template <size_t NUM_READ_OPS, size_t NUM_WRITE_OPS>
struct WomirBaseAluAdapter {
    MemoryAuxColsFactory mem_helper;
    BitwiseOperationLookup bitwise_lookup;

    template <typename T>
    using Cols = WomirBaseAluAdapterCols<T, NUM_READ_OPS, NUM_WRITE_OPS>;
    using Record = WomirBaseAluAdapterRecord<NUM_READ_OPS, NUM_WRITE_OPS>;

    __device__ WomirBaseAluAdapter(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup lookup,
        uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits), bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, Record record) {
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Cols, from_state.fp, record.fp);
        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);

        COL_WRITE_VALUE(row, Cols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(row, Cols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, Cols, rs2, record.rs2);
        COL_WRITE_VALUE(row, Cols, rs2_as, record.rs2_as);

        // Read auxiliary for fp (at from_timestamp + 0)
        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, fp_read_aux)),
            record.fp_read_aux.prev_timestamp,
            record.from_timestamp
        );

        // rs1 reads (at from_timestamp + 1 + r for r in 0..NUM_READ_OPS)
        constexpr size_t read_aux_elem_size = sizeof(MemoryReadAuxCols<uint8_t>);
#pragma unroll
        for (size_t r = 0; r < NUM_READ_OPS; r++) {
            mem_helper.fill(
                row.slice_from(
                    offsetof(Cols<uint8_t>, rs1_reads_aux) + r * read_aux_elem_size
                ),
                record.rs1_reads_aux[r].prev_timestamp,
                record.from_timestamp + 1 + r
            );
        }

        // rs2: register read when rs2_as != 0, otherwise immediate.
        if (record.rs2_as != 0) {
#pragma unroll
            for (size_t r = 0; r < NUM_READ_OPS; r++) {
                mem_helper.fill(
                    row.slice_from(
                        offsetof(Cols<uint8_t>, rs2_reads_aux) + r * read_aux_elem_size
                    ),
                    record.rs2_reads_aux[r].prev_timestamp,
                    record.from_timestamp + 1 + NUM_READ_OPS + r
                );
            }
        } else {
#pragma unroll
            for (size_t r = 0; r < NUM_READ_OPS; r++) {
                RowSlice rs2_aux = row.slice_from(
                    offsetof(Cols<uint8_t>, rs2_reads_aux) + r * read_aux_elem_size
                );
#pragma unroll
                for (size_t i = 0; i < read_aux_elem_size; i++) {
                    rs2_aux.write(i, 0);
                }
            }
            uint32_t mask = (1u << RV32_CELL_BITS) - 1u;
            bitwise_lookup.add_range(record.rs2 & mask, (record.rs2 >> RV32_CELL_BITS) & mask);
        }

        // Writes (at from_timestamp + 1 + 2*NUM_READ_OPS + w for w in 0..NUM_WRITE_OPS)
        // Type alias avoids commas inside offsetof/sizeof macros (preprocessor limitation).
        using WriteAuxCols = MemoryWriteAuxCols<uint8_t, RV32_REGISTER_NUM_LIMBS>;
        constexpr size_t write_aux_elem_size = sizeof(WriteAuxCols);
        constexpr size_t prev_data_offset = offsetof(WriteAuxCols, prev_data);
#pragma unroll
        for (size_t w = 0; w < NUM_WRITE_OPS; w++) {
            size_t base = offsetof(Cols<uint8_t>, writes_aux) + w * write_aux_elem_size;
            row.write_array(
                base + prev_data_offset,
                RV32_REGISTER_NUM_LIMBS,
                record.writes_aux[w].prev_data
            );
            mem_helper.fill(
                row.slice_from(base),
                record.writes_aux[w].prev_timestamp,
                record.from_timestamp + 1 + 2 * NUM_READ_OPS + w
            );
        }
    }
};

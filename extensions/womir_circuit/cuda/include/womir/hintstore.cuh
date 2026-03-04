// CUDA tracegen for HintStore chip.
// Multi-row chip: each instruction generates num_words rows.
// No adapter+core split: unified structure (like Const32).
#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"
#include "womir/execution.cuh"

using namespace riscv;

// Record header layout must match Rust Rv32HintStoreRecordHeader exactly (repr(C)).
struct HintStoreRecordHeader {
    uint32_t num_words;
    uint32_t from_pc;
    uint32_t timestamp;
    uint32_t fp;
    MemoryReadAuxRecord fp_read_aux;
    uint32_t mem_ptr_ptr;
    uint32_t mem_ptr;
    MemoryReadAuxRecord mem_ptr_aux_record;
    uint32_t num_words_ptr;
    MemoryReadAuxRecord num_words_read;
};

// Variable-length part of the record, one per word written.
// Must match Rust Rv32HintStoreVar exactly (repr(C)).
struct HintStoreVar {
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> data_write_aux;
    uint8_t data[RV32_REGISTER_NUM_LIMBS];
};

// Column layout must match Rust Rv32HintStoreCols exactly.
template <typename T>
struct HintStoreCols {
    // common
    T is_single;
    T is_buffer;
    T rem_words_limbs[RV32_REGISTER_NUM_LIMBS];

    WomirExecutionState<T> from_state;
    MemoryReadAuxCols<T> fp_read_aux;
    T mem_ptr_ptr;
    T mem_ptr_limbs[RV32_REGISTER_NUM_LIMBS];
    MemoryReadAuxCols<T> mem_ptr_aux_cols;

    MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS> write_aux;
    T data[RV32_REGISTER_NUM_LIMBS];

    // only buffer
    T is_buffer_start;
    T num_words_ptr;
    MemoryReadAuxCols<T> num_words_aux_cols;
};

struct HintStoreTraceFiller {
    MemoryAuxColsFactory mem_helper;
    BitwiseOperationLookup bitwise_lookup;
    uint32_t pointer_max_bits;

    template <typename T>
    using Cols = HintStoreCols<T>;

    __device__ HintStoreTraceFiller(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup lookup,
        uint32_t timestamp_max_bits,
        uint32_t pointer_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits),
          bitwise_lookup(lookup),
          pointer_max_bits(pointer_max_bits) {}

    // Fill all rows for one HintStore instruction.
    // `trace` points to the first row of this instruction in the trace matrix.
    // The instruction occupies `num_words` consecutive rows.
    __device__ void fill_rows(
        Fp *trace,
        size_t height,
        HintStoreRecordHeader const &header,
        HintStoreVar const *vars
    ) {
        uint32_t num_words = header.num_words;
        bool is_single = (header.num_words_ptr == UINT32_MAX);

        uint32_t msl_rshift = (RV32_REGISTER_NUM_LIMBS - 1) * RV32_CELL_BITS;
        uint32_t msl_lshift = RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - pointer_max_bits;

        // Range-check mem_ptr and num_words most significant limbs
        bitwise_lookup.add_range(
            (header.mem_ptr >> msl_rshift) << msl_lshift,
            (num_words >> msl_rshift) << msl_lshift
        );

        // Fill rows in forward order (idx 0 = first word)
        uint32_t timestamp = header.timestamp;
        uint32_t mem_ptr = header.mem_ptr;

        for (uint32_t idx = 0; idx < num_words; idx++) {
            RowSlice row(trace + (idx), height);
            auto const &var = vars[idx];

            // Range-check data bytes
            for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS / 2; i++) {
                bitwise_lookup.add_range(var.data[2 * i], var.data[2 * i + 1]);
            }

            COL_WRITE_VALUE(row, Cols, is_single, is_single ? 1 : 0);
            COL_WRITE_VALUE(row, Cols, is_buffer, is_single ? 0 : 1);

            // rem_words_limbs: (num_words - idx) as LE bytes
            uint32_t rem_words = num_words - idx;
            uint8_t rem_words_bytes[RV32_REGISTER_NUM_LIMBS];
            for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
                rem_words_bytes[i] = (rem_words >> (8 * i)) & 0xFF;
            }
            COL_WRITE_ARRAY(row, Cols, rem_words_limbs, rem_words_bytes);

            // from_state
            COL_WRITE_VALUE(row, Cols, from_state.pc, header.from_pc);
            COL_WRITE_VALUE(row, Cols, from_state.fp, header.fp);
            COL_WRITE_VALUE(row, Cols, from_state.timestamp, timestamp);

            // mem_ptr_ptr
            COL_WRITE_VALUE(row, Cols, mem_ptr_ptr, header.mem_ptr_ptr);

            // mem_ptr_limbs
            uint8_t mem_ptr_bytes[RV32_REGISTER_NUM_LIMBS];
            for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
                mem_ptr_bytes[i] = (mem_ptr >> (8 * i)) & 0xFF;
            }
            COL_WRITE_ARRAY(row, Cols, mem_ptr_limbs, mem_ptr_bytes);

            // data
            COL_WRITE_ARRAY(row, Cols, data, var.data);

            // write_aux: set prev_data and fill timestamp proof
            // Write happens at timestamp + 3 (fp_read=+0, mem_ptr_read=+1, num_words_read=+2, write=+3)
            using WriteAuxCols = MemoryWriteAuxCols<uint8_t, RV32_REGISTER_NUM_LIMBS>;
            size_t write_base = COL_INDEX(Cols, write_aux);
            row.write_array(
                write_base + offsetof(WriteAuxCols, prev_data),
                RV32_REGISTER_NUM_LIMBS,
                var.data_write_aux.prev_data
            );
            mem_helper.fill(
                row.slice_from(write_base),
                var.data_write_aux.prev_timestamp,
                timestamp + 3
            );

            if (idx == 0) {
                // First row: fill fp_read_aux, mem_ptr_aux_cols
                mem_helper.fill(
                    row.slice_from(COL_INDEX(Cols, fp_read_aux)),
                    header.fp_read_aux.prev_timestamp,
                    timestamp
                );
                mem_helper.fill(
                    row.slice_from(COL_INDEX(Cols, mem_ptr_aux_cols)),
                    header.mem_ptr_aux_record.prev_timestamp,
                    timestamp + 1
                );

                if (!is_single) {
                    // Buffer start: fill num_words_aux_cols
                    COL_WRITE_VALUE(row, Cols, is_buffer_start, 1);
                    COL_WRITE_VALUE(row, Cols, num_words_ptr, header.num_words_ptr);
                    mem_helper.fill(
                        row.slice_from(COL_INDEX(Cols, num_words_aux_cols)),
                        header.num_words_read.prev_timestamp,
                        timestamp + 2
                    );
                } else {
                    COL_WRITE_VALUE(row, Cols, is_buffer_start, 0);
                    COL_WRITE_VALUE(row, Cols, num_words_ptr, 0);
                    mem_helper.fill_zero(
                        row.slice_from(COL_INDEX(Cols, num_words_aux_cols))
                    );
                }
            } else {
                // Non-first rows: zero out read auxs
                mem_helper.fill_zero(
                    row.slice_from(COL_INDEX(Cols, fp_read_aux))
                );
                mem_helper.fill_zero(
                    row.slice_from(COL_INDEX(Cols, mem_ptr_aux_cols))
                );
                COL_WRITE_VALUE(row, Cols, is_buffer_start, 0);
                COL_WRITE_VALUE(row, Cols, num_words_ptr, 0);
                mem_helper.fill_zero(
                    row.slice_from(COL_INDEX(Cols, num_words_aux_cols))
                );
            }

            // Advance for next row
            mem_ptr += RV32_REGISTER_NUM_LIMBS;
            timestamp += 4;
        }
    }
};

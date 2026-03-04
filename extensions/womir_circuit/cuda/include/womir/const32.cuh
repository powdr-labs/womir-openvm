// CUDA tracegen for Const32 chip.
// Unlike ALU chips, Const32 has no adapter+core split: it's a single unified structure.
#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"
#include "womir/execution.cuh"

using namespace riscv;

// Record layout must match Rust Const32Record exactly (repr(C)).
struct Const32Record {
    uint32_t from_pc;
    uint32_t fp;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t imm;
    MemoryReadAuxRecord fp_read_aux;
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> writes_aux;
};

// Column layout must match Rust Const32AdapterAirCol exactly.
template <typename T, size_t NUM_LIMBS = RV32_REGISTER_NUM_LIMBS>
struct Const32Cols {
    T is_valid;
    WomirExecutionState<T> from_state;
    T rd_ptr;
    T imm_limbs[NUM_LIMBS];
    MemoryReadAuxCols<T> fp_read_aux;
    MemoryWriteAuxCols<T, NUM_LIMBS> write_aux;
};

struct Const32TraceFiller {
    MemoryAuxColsFactory mem_helper;
    BitwiseOperationLookup bitwise_lookup;

    template <typename T, size_t NUM_LIMBS = RV32_REGISTER_NUM_LIMBS>
    using Cols = Const32Cols<T, NUM_LIMBS>;

    __device__ Const32TraceFiller(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup lookup,
        uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits), bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, Const32Record const &record) {
        // is_valid
        COL_WRITE_VALUE(row, Cols, is_valid, 1);

        // from_state
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Cols, from_state.fp, record.fp);
        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);

        // rd_ptr
        COL_WRITE_VALUE(row, Cols, rd_ptr, record.rd_ptr);

        // imm_limbs: decompose the 32-bit immediate into 8-bit limbs
        uint32_t imm = record.imm;
        constexpr uint32_t mask = (1u << RV32_CELL_BITS) - 1u;
        uint8_t imm_limbs[RV32_REGISTER_NUM_LIMBS];
#pragma unroll
        for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
            imm_limbs[i] = (imm >> (RV32_CELL_BITS * i)) & mask;
        }
        COL_WRITE_ARRAY(row, Cols, imm_limbs, imm_limbs);

        // Range-check imm_limbs via bitwise lookup (pairs)
        bitwise_lookup.add_range(imm_limbs[0], imm_limbs[1]);
        bitwise_lookup.add_range(imm_limbs[2], imm_limbs[3]);

        // fp_read_aux: fill timestamp proof for FP read at from_timestamp + 0
        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, fp_read_aux)),
            record.fp_read_aux.prev_timestamp,
            record.from_timestamp
        );

        // write_aux: set prev_data and fill timestamp proof
        // Write happens at from_timestamp + 1 (after FP read at from_timestamp + 0)
        using WriteAuxCols = MemoryWriteAuxCols<uint8_t, RV32_REGISTER_NUM_LIMBS>;
        size_t write_base = COL_INDEX(Cols, write_aux);
        row.write_array(
            write_base + offsetof(WriteAuxCols, prev_data),
            RV32_REGISTER_NUM_LIMBS,
            record.writes_aux.prev_data
        );
        mem_helper.fill(
            row.slice_from(write_base),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 1
        );
    }
};

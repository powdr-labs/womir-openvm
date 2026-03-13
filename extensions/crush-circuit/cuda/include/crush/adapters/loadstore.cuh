// Adapted from <openvm>/extensions/rv32im/circuit/cuda/include/rv32im/adapters/loadstore.cuh
// Main changes: adds frame pointer (fp) field, CrushExecutionState, fp_read_aux, timestamp +1 shift,
// imm_lo/imm_hi instead of imm/imm_sign
#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "crush/execution.cuh"

using namespace riscv;

template <typename T> struct CrushLoadStoreAdapterCols {
    CrushExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[RV32_REGISTER_NUM_LIMBS];
    MemoryReadAuxCols<T> fp_read_aux;
    MemoryReadAuxCols<T> rs1_aux_cols;

    /// Will write to rd when Load and read from rs2 when Store
    T rd_rs2_ptr;
    MemoryReadAuxCols<T> read_data_aux;
    T imm_lo;
    T imm_hi;
    /// mem_ptr is the intermediate memory pointer limbs, needed to check the correct addition
    T mem_ptr_limbs[2];
    T mem_as;
    /// prev_data will be provided by the core chip to make a complete MemoryWriteAuxCols
    MemoryBaseAuxCols<T> write_base_aux;
    /// Only writes if `needs_write`.
    T needs_write;
};

struct CrushLoadStoreAdapterRecord {
    uint32_t from_pc;
    uint32_t fp;
    uint32_t from_timestamp;

    uint32_t rs1_ptr;
    uint32_t rs1_val;
    MemoryReadAuxRecord fp_read_aux;
    MemoryReadAuxRecord rs1_aux_record;

    uint32_t rd_rs2_ptr;
    MemoryReadAuxRecord read_data_aux;
    uint16_t imm_lo;
    uint16_t imm_hi;

    uint8_t mem_as;

    uint32_t write_prev_timestamp;
};

struct CrushLoadStoreAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ CrushLoadStoreAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, CrushLoadStoreAdapterRecord record) {
        COL_WRITE_VALUE(row, CrushLoadStoreAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, CrushLoadStoreAdapterCols, from_state.fp, record.fp);
        COL_WRITE_VALUE(row, CrushLoadStoreAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, CrushLoadStoreAdapterCols, rs1_ptr, record.rs1_ptr);

        auto rs1_data = reinterpret_cast<uint8_t *>(&record.rs1_val);
        COL_WRITE_ARRAY(row, CrushLoadStoreAdapterCols, rs1_data, rs1_data);

        bool needs_write = record.rd_rs2_ptr != UINT32_MAX;

        // Read fp (at from_timestamp + 0)
        mem_helper.fill(
            row.slice_from(COL_INDEX(CrushLoadStoreAdapterCols, fp_read_aux)),
            record.fp_read_aux.prev_timestamp,
            record.from_timestamp
        );

        // Read rs1 (at from_timestamp + 1)
        mem_helper.fill(
            row.slice_from(COL_INDEX(CrushLoadStoreAdapterCols, rs1_aux_cols)),
            record.rs1_aux_record.prev_timestamp,
            record.from_timestamp + 1
        );

        if (needs_write) {
            COL_WRITE_VALUE(row, CrushLoadStoreAdapterCols, rd_rs2_ptr, record.rd_rs2_ptr);
        } else {
            COL_WRITE_VALUE(row, CrushLoadStoreAdapterCols, rd_rs2_ptr, 0);
        }

        // Read data (at from_timestamp + 2)
        mem_helper.fill(
            row.slice_from(COL_INDEX(CrushLoadStoreAdapterCols, read_data_aux)),
            record.read_data_aux.prev_timestamp,
            record.from_timestamp + 2
        );

        COL_WRITE_VALUE(row, CrushLoadStoreAdapterCols, imm_lo, record.imm_lo);
        COL_WRITE_VALUE(row, CrushLoadStoreAdapterCols, imm_hi, record.imm_hi);

        uint32_t ptr = record.rs1_val + ((uint32_t)record.imm_lo | ((uint32_t)record.imm_hi << 16));
        auto ptr_limbs = reinterpret_cast<uint16_t *>(&ptr);
        COL_WRITE_ARRAY(row, CrushLoadStoreAdapterCols, mem_ptr_limbs, ptr_limbs);
        COL_WRITE_VALUE(row, CrushLoadStoreAdapterCols, mem_as, record.mem_as);

        range_checker.add_count((uint32_t)ptr_limbs[0] >> 2, RV32_CELL_BITS * 2 - 2);
        range_checker.add_count((uint32_t)ptr_limbs[1], pointer_max_bits - 16);

        COL_WRITE_VALUE(row, CrushLoadStoreAdapterCols, needs_write, needs_write);
        if (needs_write) {
            // Write (at from_timestamp + 3)
            mem_helper.fill(
                row.slice_from(COL_INDEX(CrushLoadStoreAdapterCols, write_base_aux)),
                record.write_prev_timestamp,
                record.from_timestamp + 3
            );
        } else {
            mem_helper.fill_zero(
                row.slice_from(COL_INDEX(CrushLoadStoreAdapterCols, write_base_aux))
            );
        }
    }
};

// CUDA tracegen for HintStore chip (multi-row).
// One thread per instruction; each thread fills num_words consecutive rows.
#include "launcher.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"
#include "womir/execution.cuh"

using namespace riscv;

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

// This is the part of the record that we keep only once per instruction.
// Must match Rust Rv32HintStoreRecordHeader exactly (repr(C)).
struct HintStoreRecordHeader {
    uint32_t num_words;
    uint32_t from_pc;
    uint32_t timestamp;
    uint32_t fp;
    MemoryReadAuxRecord fp_read_aux;
    uint32_t mem_ptr_ptr;
    uint32_t mem_ptr;
    MemoryReadAuxRecord mem_ptr_aux_record;
    // will set `num_words_ptr` to `u32::MAX` in case of single hint
    uint32_t num_words_ptr;
    MemoryReadAuxRecord num_words_read;
};

// This is the part of the record that we keep `num_words` times per instruction.
// Must match Rust Rv32HintStoreVar exactly (repr(C)).
struct HintStoreVar {
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> data_write_aux;
    uint8_t data[RV32_REGISTER_NUM_LIMBS];
};

struct HintStoreTraceFiller {
    size_t pointer_max_bits;
    BitwiseOperationLookup bitwise_lookup;
    MemoryAuxColsFactory mem_helper;

    __device__ HintStoreTraceFiller(
        BitwiseOperationLookup bitwise_lookup,
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : bitwise_lookup(bitwise_lookup), pointer_max_bits(pointer_max_bits),
          mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(
        RowSlice row,
        HintStoreRecordHeader &record,
        HintStoreVar &write,
        uint32_t local_idx
    ) {
        bool is_single = record.num_words_ptr == UINT32_MAX;
        // Timestamp delta is 4 per row: fp_read=+0, mem_ptr_read=+1, num_words_read=+2, write=+3
        uint32_t timestamp = record.timestamp + local_idx * 4;
        uint32_t rem_words = record.num_words - local_idx;
        uint32_t mem_ptr = record.mem_ptr + local_idx * (uint32_t)RV32_REGISTER_NUM_LIMBS;
        auto rem_words_limbs = reinterpret_cast<uint8_t *>(&rem_words);
        auto mem_ptr_limbs = reinterpret_cast<uint8_t *>(&mem_ptr);

        COL_WRITE_VALUE(row, HintStoreCols, is_single, is_single);
        COL_WRITE_VALUE(row, HintStoreCols, is_buffer, !is_single);
        COL_WRITE_ARRAY(row, HintStoreCols, rem_words_limbs, rem_words_limbs);
        COL_WRITE_VALUE(row, HintStoreCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, HintStoreCols, from_state.fp, record.fp);
        COL_WRITE_VALUE(row, HintStoreCols, from_state.timestamp, timestamp);
        COL_WRITE_VALUE(row, HintStoreCols, mem_ptr_ptr, record.mem_ptr_ptr);
        COL_WRITE_ARRAY(row, HintStoreCols, mem_ptr_limbs, mem_ptr_limbs);

        if (local_idx == 0) {
            uint32_t msl_rshift = (RV32_REGISTER_NUM_LIMBS - 1) * RV32_CELL_BITS;
            uint32_t msl_lshift = RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - pointer_max_bits;
            bitwise_lookup.add_range(
                (record.mem_ptr >> msl_rshift) << msl_lshift,
                (record.num_words >> msl_rshift) << msl_lshift
            );
            // fp_read at timestamp + 0
            mem_helper.fill(
                row.slice_from(COL_INDEX(HintStoreCols, fp_read_aux)),
                record.fp_read_aux.prev_timestamp,
                timestamp
            );
            // mem_ptr_read at timestamp + 1
            mem_helper.fill(
                row.slice_from(COL_INDEX(HintStoreCols, mem_ptr_aux_cols)),
                record.mem_ptr_aux_record.prev_timestamp,
                timestamp + 1
            );
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(HintStoreCols, fp_read_aux)));
            mem_helper.fill_zero(row.slice_from(COL_INDEX(HintStoreCols, mem_ptr_aux_cols)));
        }

        if (local_idx == 0 && !is_single) {
            // num_words_read at timestamp + 2
            mem_helper.fill(
                row.slice_from(COL_INDEX(HintStoreCols, num_words_aux_cols)),
                record.num_words_read.prev_timestamp,
                timestamp + 2
            );
            COL_WRITE_VALUE(row, HintStoreCols, is_buffer_start, 1);
            COL_WRITE_VALUE(row, HintStoreCols, num_words_ptr, record.num_words_ptr);
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(HintStoreCols, num_words_aux_cols)));
            COL_WRITE_VALUE(row, HintStoreCols, is_buffer_start, 0);
            COL_WRITE_VALUE(row, HintStoreCols, num_words_ptr, 0);
        }

        COL_WRITE_ARRAY(row, HintStoreCols, write_aux.prev_data, write.data_write_aux.prev_data);
        // write at timestamp + 3
        mem_helper.fill(
            row.slice_from(COL_INDEX(HintStoreCols, write_aux)),
            write.data_write_aux.prev_timestamp,
            timestamp + 3
        );

        COL_WRITE_ARRAY(row, HintStoreCols, data, write.data);
#pragma unroll
        for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i += 2) {
            bitwise_lookup.add_range(write.data[i], write.data[i + 1]);
        }
    }
};

// Combined kernel: threads [0, num_instructions) fill trace rows from records,
// threads [num_instructions, height) zero-fill padding rows.
__global__ void womir_hintstore_tracegen(
    Fp *d_trace,
    size_t height,
    uint8_t const *d_records,
    uint32_t const *d_record_offsets,
    uint32_t const *d_row_offsets,
    uint32_t num_instructions,
    uint32_t total_rows,
    uint32_t pointer_max_bits,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_instructions) {
        uint32_t rec_offset = d_record_offsets[idx];
        uint32_t row_offset = d_row_offsets[idx];

        auto record = *reinterpret_cast<HintStoreRecordHeader const *>(d_records + rec_offset);

        constexpr size_t header_size = sizeof(HintStoreRecordHeader);
        constexpr size_t var_align = alignof(HintStoreVar);
        constexpr size_t aligned_header = (header_size + var_align - 1) & ~(var_align - 1);
        auto const *vars = reinterpret_cast<HintStoreVar const *>(d_records + rec_offset + aligned_header);

        auto filler = HintStoreTraceFiller(
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            pointer_max_bits,
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );

        for (uint32_t local_idx = 0; local_idx < record.num_words; local_idx++) {
            RowSlice row(d_trace + (row_offset + local_idx), height);
            auto data_write = vars[local_idx];
            filler.fill_trace_row(row, record, data_write, local_idx);
        }
    }

    // Zero-fill padding rows
    uint32_t num_padding = height - total_rows;
    if (idx < num_padding) {
        RowSlice row(d_trace + (total_rows + idx), height);
        row.fill_zero(0, sizeof(HintStoreCols<uint8_t>));
    }
}

extern "C" int _womir_hintstore_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t const *d_records,
    uint32_t const *d_record_offsets,
    uint32_t const *d_row_offsets,
    uint32_t num_instructions,
    uint32_t total_rows,
    uint32_t pointer_max_bits,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= total_rows);
    assert(width == sizeof(HintStoreCols<uint8_t>));

    uint32_t num_padding = height - total_rows;
    uint32_t num_threads = num_instructions > num_padding ? num_instructions : num_padding;
    if (num_threads == 0) {
        return 0;
    }
    auto [grid, block] = kernel_launch_params(num_threads);
    womir_hintstore_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_record_offsets,
        d_row_offsets,
        num_instructions,
        total_rows,
        pointer_max_bits,
        d_range_checker_ptr,
        range_checker_num_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

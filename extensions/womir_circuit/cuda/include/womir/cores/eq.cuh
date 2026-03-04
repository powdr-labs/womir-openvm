#pragma once

#include "primitives/constants.h"
#include "primitives/trace_access.h"

using namespace riscv;

template <size_t NUM_LIMBS>
struct EqCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS>
struct EqCoreCols {
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T cmp_result;
    T opcode_eq_flag;
    T opcode_neq_flag;
    T diff_inv_marker[NUM_LIMBS];
};

template <size_t NUM_LIMBS>
struct EqCore {
    template <typename T> using Cols = EqCoreCols<T, NUM_LIMBS>;

    __device__ void fill_trace_row(RowSlice row, EqCoreRecord<NUM_LIMBS> record) {
        constexpr uint8_t EQ_OPCODE = 0;

        bool is_eq = record.local_opcode == EQ_OPCODE;

        // Find the first differing position and compute field inverse
        bool values_equal = true;
        size_t diff_idx = 0;
        Fp diff_inv_val = Fp::zero();

#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            if (values_equal && record.b[i] != record.c[i]) {
                values_equal = false;
                diff_idx = i;
                diff_inv_val = inv(Fp(record.b[i]) - Fp(record.c[i]));
            }
        }

        bool cmp_result = values_equal ? is_eq : !is_eq;

        // Write diff_inv_marker array: all zeros except at diff_idx
        Fp diff_inv_marker[NUM_LIMBS];
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            diff_inv_marker[i] = Fp::zero();
        }
        if (!values_equal) {
            diff_inv_marker[diff_idx] = diff_inv_val;
        }

        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);
        COL_WRITE_ARRAY(row, Cols, diff_inv_marker, diff_inv_marker);

        COL_WRITE_VALUE(row, Cols, cmp_result, cmp_result);
        COL_WRITE_VALUE(row, Cols, opcode_eq_flag, is_eq);
        COL_WRITE_VALUE(row, Cols, opcode_neq_flag, !is_eq);
    }
};

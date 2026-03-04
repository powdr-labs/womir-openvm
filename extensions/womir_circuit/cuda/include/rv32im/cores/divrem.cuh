#pragma once

// Adapted from <openvm>/extensions/rv32im/circuit/cuda/src/divrem.cu
// Extracted into a header and generalized for NUM_LIMBS > 4 (64-bit support).
// Diff: compare with the inlined core in OpenVM's divrem.cu.

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"

using namespace riscv;

template <typename T, size_t NUM_LIMBS> struct DivRemCoreCols {
    // b = c * q + r for some 0 <= |r| < |c | and sign(r) = sign(b) or r = 0.
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T q[NUM_LIMBS];
    T r[NUM_LIMBS];

    // Flags to indicate special cases.
    T zero_divisor;
    T r_zero;

    // Sign of b and c respectively, while q_sign = b_sign ^ c_sign if q is non-zero
    // and is 0 otherwise. sign_xor = b_sign ^ c_sign always.
    T b_sign;
    T c_sign;
    T q_sign;
    T sign_xor;

    // Auxiliary columns to constrain that zero_divisor = 1 if and only if c = 0.
    T c_sum_inv;
    // Auxiliary columns to constrain that r_zero = 1 if and only if r = 0 and zero_divisor = 0.
    T r_sum_inv;

    // Auxiliary columns to constrain that 0 <= |r| < |c|. When sign_xor == 1 we have
    // r_prime = -r, and when sign_xor == 0 we have r_prime = r. Each r_inv[i] is the
    // field inverse of r_prime[i] - 2^RV32_CELL_BITS, ensures each r_prime[i] is in range.
    T r_prime[NUM_LIMBS];
    T r_inv[NUM_LIMBS];
    T lt_marker[NUM_LIMBS];
    T lt_diff;

    // Opcode flags
    T opcode_div_flag;
    T opcode_divu_flag;
    T opcode_rem_flag;
    T opcode_remu_flag;
};

template <size_t NUM_LIMBS> struct DivRemCoreRecords {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

enum DivRemOpcode {
    DIV,
    DIVU,
    REM,
    REMU,
};

// Helper: reconstruct a value from little-endian bytes.
// Returns uint64_t for any NUM_LIMBS (up to 8).
template <size_t NUM_LIMBS>
__device__ __forceinline__ uint64_t value_from_bytes_le(const uint8_t *bytes) {
    uint64_t val = 0;
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        val |= (uint64_t)bytes[i] << (i * 8);
    }
    return val;
}

// Helper: compute absolute value for two's complement with the given number of limbs.
template <size_t NUM_LIMBS>
__device__ __forceinline__ uint64_t abs_val(uint64_t x, bool is_neg) {
    if (!is_neg) return x;
    // Mask to NUM_LIMBS * 8 bits. For NUM_LIMBS=8 the mask is UINT64_MAX.
    constexpr uint64_t MASK = (NUM_LIMBS < 8) ? ((1ULL << (NUM_LIMBS * 8)) - 1) : UINT64_MAX;
    return (MASK - x + 1) & MASK;
}

template <size_t NUM_LIMBS> struct DivRemCore {
    BitwiseOperationLookup bitwise_lookup;
    RangeTupleChecker<2> range_tuple_checker;

    template <typename T> using Cols = DivRemCoreCols<T, NUM_LIMBS>;

    __device__ DivRemCore(
        BitwiseOperationLookup bitwise_lookup,
        RangeTupleChecker<2> range_tuple_checker
    )
        : bitwise_lookup(bitwise_lookup), range_tuple_checker(range_tuple_checker) {}

    __device__ void fill_trace_row(RowSlice row, DivRemCoreRecords<NUM_LIMBS> const &record) {
        DivRemOpcode opcode = static_cast<DivRemOpcode>(record.local_opcode);

        bool is_signed = opcode == DIV || opcode == REM;
        bool b_sign = is_signed && (record.b[NUM_LIMBS - 1] >> 7);
        bool c_sign = is_signed && (record.c[NUM_LIMBS - 1] >> 7);
        bool q_sign = false;
        bool case_none = false;

        uint64_t b_val = value_from_bytes_le<NUM_LIMBS>(record.b);
        uint64_t c_val = value_from_bytes_le<NUM_LIMBS>(record.c);
        uint64_t q_val = 0;
        uint64_t r_val = 0;

        constexpr uint64_t MAX_VAL = (NUM_LIMBS < 8)
            ? ((1ULL << (NUM_LIMBS * 8)) - 1) : UINT64_MAX;
        constexpr uint64_t MIN_SIGNED = 1ULL << (NUM_LIMBS * 8 - 1);

        if (c_val == 0) {
            q_val = MAX_VAL;
            r_val = b_val;
            q_sign = is_signed;
        } else if ((b_val == MIN_SIGNED) && (c_val == MAX_VAL) && b_sign && c_sign) {
            q_val = b_val;
            r_val = 0;
            q_sign = false;
        } else {
            uint64_t b_abs = abs_val<NUM_LIMBS>(b_val, b_sign);
            uint64_t c_abs = abs_val<NUM_LIMBS>(c_val, c_sign);
            q_val = abs_val<NUM_LIMBS>(b_abs / c_abs, b_sign != c_sign);
            r_val = abs_val<NUM_LIMBS>(b_abs % c_abs, b_sign);
            q_sign = is_signed && (q_val >> (NUM_LIMBS * 8 - 1));
            case_none = true;
        }

        uint8_t q[NUM_LIMBS];
        uint8_t r[NUM_LIMBS];
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            q[i] = (uint8_t)(q_val >> (i * 8));
            r[i] = (uint8_t)(r_val >> (i * 8));
        }

        uint64_t r_prime_val = abs_val<NUM_LIMBS>(r_val, b_sign ^ c_sign);
        uint8_t r_prime[NUM_LIMBS];
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            r_prime[i] = (uint8_t)(r_prime_val >> (i * 8));
        }
        bool r_zero = (r_val == 0) && (c_val != 0);

        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);
        COL_WRITE_ARRAY(row, Cols, q, q);
        COL_WRITE_ARRAY(row, Cols, r, r);
        COL_WRITE_VALUE(row, Cols, zero_divisor, c_val == 0);
        COL_WRITE_VALUE(row, Cols, r_zero, r_zero);
        COL_WRITE_VALUE(row, Cols, b_sign, b_sign);
        COL_WRITE_VALUE(row, Cols, c_sign, c_sign);
        COL_WRITE_VALUE(row, Cols, q_sign, q_sign);
        COL_WRITE_VALUE(row, Cols, sign_xor, b_sign ^ c_sign);

        uint32_t c_sum = 0;
        uint32_t r_sum = 0;
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            c_sum += record.c[i];
            r_sum += r[i];
        }
        if (c_sum == 0) {
            COL_WRITE_VALUE(row, Cols, c_sum_inv, 0);
        } else {
            COL_WRITE_VALUE(row, Cols, c_sum_inv, inv(Fp(c_sum)));
        }

        if (r_sum == 0) {
            COL_WRITE_VALUE(row, Cols, r_sum_inv, 0);
        } else {
            COL_WRITE_VALUE(row, Cols, r_sum_inv, inv(Fp(r_sum)));
        }

        COL_WRITE_ARRAY(row, Cols, r_prime, r_prime);
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            Fp r_inv = inv(Fp(r_prime[i]) - Fp(256));
            COL_WRITE_VALUE(row, Cols, r_inv[i], r_inv);
        }

        COL_FILL_ZERO(row, Cols, lt_marker);
        if (case_none && !r_zero) {
            uint32_t idx = NUM_LIMBS;
#pragma unroll
            for (int i = NUM_LIMBS - 1; i >= 0; i--) {
                if (record.c[i] != r_prime[i]) {
                    idx = i;
                    break;
                }
            }
            uint8_t val = 0;
            if (c_sign) {
                val = r_prime[idx] - record.c[idx];
            } else {
                val = record.c[idx] - r_prime[idx];
            }
            bitwise_lookup.add_range(val - 1, 0);
            COL_WRITE_VALUE(row, Cols, lt_marker[idx], 1);
            COL_WRITE_VALUE(row, Cols, lt_diff, val);
        } else {
            COL_WRITE_VALUE(row, Cols, lt_diff, 0);
        }

        COL_WRITE_VALUE(row, Cols, opcode_div_flag, opcode == DIV);
        COL_WRITE_VALUE(row, Cols, opcode_divu_flag, opcode == DIVU);
        COL_WRITE_VALUE(row, Cols, opcode_rem_flag, opcode == REM);
        COL_WRITE_VALUE(row, Cols, opcode_remu_flag, opcode == REMU);

        if (is_signed) {
            bitwise_lookup.add_range(
                (record.b[NUM_LIMBS - 1] & 0x7f) << 1, (record.c[NUM_LIMBS - 1] & 0x7f) << 1
            );
        }

        // range tuple check carries
        uint32_t carry = 0;
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            carry += r[i];
#pragma unroll
            for (size_t j = 0; j <= i; j++) {
                carry += (uint32_t)q[j] * (uint32_t)record.c[i - j];
            }
            carry = carry >> RV32_CELL_BITS;
            range_tuple_checker.add_count((uint32_t[2]){(uint32_t)q[i], carry});
        }
        bool r_sign = is_signed && (r[NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1));

        uint32_t q_ext = (q_sign && is_signed) * ((1 << RV32_CELL_BITS) - 1);
        uint32_t c_ext = (c_sign << RV32_CELL_BITS) - c_sign;
        uint32_t r_ext = (r_sign << RV32_CELL_BITS) - r_sign;

        uint32_t c_pref = 0;
        uint32_t q_pref = 0;
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            c_pref += record.c[i];
            q_pref += q[i];
            carry += c_pref * q_ext + q_pref * c_ext + r_ext;
#pragma unroll
            for (size_t j = i + 1; j < NUM_LIMBS; j++) {
                carry += (uint32_t)record.c[j] * (uint32_t)q[NUM_LIMBS + i - j];
            }
            carry = carry >> RV32_CELL_BITS;
            range_tuple_checker.add_count((uint32_t[2]){(uint32_t)r[i], carry});
        }
    }
};

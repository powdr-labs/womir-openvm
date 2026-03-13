// WOMIR Jump core for CUDA tracegen.
// Mirrors JumpCoreFiller::fill_trace_row from jump/core.rs.
#pragma once

#include "primitives/constants.h"
#include "primitives/trace_access.h"

using namespace riscv;

struct JumpCoreRecord {
    uint8_t rs_val[RV32_REGISTER_NUM_LIMBS];
    uint32_t imm;
    uint8_t local_opcode;
};

template <typename T> struct JumpCoreCols {
    T rs_val[RV32_REGISTER_NUM_LIMBS];
    T imm;
    T opcode_jump_flag;
    T opcode_skip_flag;
    T opcode_jump_if_flag;
    T opcode_jump_if_zero_flag;
    T cond_is_zero;
    T do_absolute_jump;
    T nonzero_inv_marker[RV32_REGISTER_NUM_LIMBS];
};

// Opcode indices matching JumpOpcode enum order
enum JumpOpcode : uint8_t {
    JUMP = 0,
    SKIP = 1,
    JUMP_IF = 2,
    JUMP_IF_ZERO = 3,
};

struct JumpCore {
    template <typename T> using Cols = JumpCoreCols<T>;

    __device__ void fill_trace_row(RowSlice row, JumpCoreRecord record) {
        // Compute cond_is_zero and nonzero_inv_marker
        bool cond_is_zero = true;
        Fp nonzero_inv_marker[RV32_REGISTER_NUM_LIMBS];
#pragma unroll
        for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
            nonzero_inv_marker[i] = Fp::zero();
        }
        for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
            if (record.rs_val[i] != 0) {
                cond_is_zero = false;
                nonzero_inv_marker[i] = inv(Fp(record.rs_val[i]));
                break;
            }
        }

        // Compute do_absolute_jump
        bool do_absolute_jump;
        switch (record.local_opcode) {
        case JUMP:
            do_absolute_jump = true;
            break;
        case SKIP:
            do_absolute_jump = false;
            break;
        case JUMP_IF:
            do_absolute_jump = !cond_is_zero;
            break;
        case JUMP_IF_ZERO:
            do_absolute_jump = cond_is_zero;
            break;
        default:
            do_absolute_jump = false;
        }

        // Write columns (reverse order to match CPU filler pattern)
        COL_WRITE_ARRAY(row, Cols, nonzero_inv_marker, nonzero_inv_marker);
        COL_WRITE_VALUE(row, Cols, do_absolute_jump, do_absolute_jump);
        COL_WRITE_VALUE(row, Cols, cond_is_zero, cond_is_zero);

        COL_WRITE_VALUE(row, Cols, opcode_jump_if_zero_flag, record.local_opcode == JUMP_IF_ZERO);
        COL_WRITE_VALUE(row, Cols, opcode_jump_if_flag, record.local_opcode == JUMP_IF);
        COL_WRITE_VALUE(row, Cols, opcode_skip_flag, record.local_opcode == SKIP);
        COL_WRITE_VALUE(row, Cols, opcode_jump_flag, record.local_opcode == JUMP);

        COL_WRITE_VALUE(row, Cols, imm, record.imm);
        COL_WRITE_ARRAY(row, Cols, rs_val, record.rs_val);
    }
};

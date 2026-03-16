#pragma once

#include "primitives/constants.h"

using namespace riscv;

// Number of 4-byte register operations to access a 32-bit value.
static const size_t W32_REG_OPS = 1;

// 64-bit constants
static const size_t W64_NUM_LIMBS = 2 * RV32_REGISTER_NUM_LIMBS;
static const size_t W64_REG_OPS = 2;

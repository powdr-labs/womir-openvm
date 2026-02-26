// Software float32 operations for WOMIR.
// Each function is exported individually.
// Float values are passed as unsigned 32-bit integers (bit representation).

typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;
typedef long long i64;

_Static_assert(sizeof(u32) == 4, "u32 must be 4 bytes");
_Static_assert(sizeof(u64) == 8, "u64 must be 8 bytes");

// ---- IEEE 754 constants ----
#define F32_SIGN_BIT   0x80000000u
#define F32_EXP_MASK   0x7F800000u
#define F32_FRAC_MASK  0x007FFFFFu
#define F32_EXP_BIAS   127
#define F32_EXP_SHIFT  23
#define F32_QNAN       0x7FC00000u
#define F32_INF        0x7F800000u

static inline int f32_is_nan(u32 a) {
    return (a & F32_EXP_MASK) == F32_EXP_MASK && (a & F32_FRAC_MASK) != 0;
}

static inline int f32_is_inf(u32 a) {
    return (a & 0x7FFFFFFFu) == F32_INF;
}

static inline int f32_sign(u32 a) { return (a >> 31) & 1; }
static inline int f32_exp(u32 a) { return (int)((a >> F32_EXP_SHIFT) & 0xFF); }
static inline u32 f32_frac(u32 a) { return a & F32_FRAC_MASK; }

// Round and pack a float from sign, exponent, and significand with 8 extra bits.
// sig format: [implicit_1 | 23 fraction bits | 8 guard/round/sticky bits]
// For normal numbers, the implicit 1 is at bit 31. exp is biased (1-254 for normals).
static __attribute__((always_inline)) u32 f32_round_pack(int sign, int exp, u32 sig) {
    if (exp >= 0xFF) {
        return (sign ? F32_SIGN_BIT : 0) | F32_INF;
    }
    if (exp <= 0) {
        // Subnormal: shift right to make exp=0 format
        int shift = 1 - exp;
        if (shift > 31) return sign ? F32_SIGN_BIT : 0;
        u32 sticky_bits = sig & ((1u << shift) - 1);
        sig >>= shift;
        if (sticky_bits) sig |= 1;
        exp = 0;
    }

    u32 round_bit = (sig >> 7) & 1;
    u32 sticky = sig & 0x7F;
    sig >>= 8;

    // Round to nearest even
    sig += (round_bit & (sticky | (sig & 1)));
    if (sig >= 0x01000000u) {
        sig >>= 1;
        exp++;
        if (exp >= 0xFF) return (sign ? F32_SIGN_BIT : 0) | F32_INF;
    }

    // If exp == 1 but no implicit bit set, result is subnormal
    if (exp > 0 && sig < 0x00800000u) {
        exp = 0;
    }

    return (sign ? F32_SIGN_BIT : 0) | ((u32)exp << F32_EXP_SHIFT) | (sig & F32_FRAC_MASK);
}

// ---- Internal helpers (always inlined to avoid cross-calls) ----

static __attribute__((always_inline)) u32 f32_add_impl(u32 a, u32 b) {
    if (f32_is_nan(a)) return F32_QNAN;
    if (f32_is_nan(b)) return F32_QNAN;

    int sign_a = f32_sign(a);
    int sign_b = f32_sign(b);
    int exp_a = f32_exp(a);
    int exp_b = f32_exp(b);
    u32 sig_a = f32_frac(a);
    u32 sig_b = f32_frac(b);

    if (exp_a == 0xFF) {
        if (exp_b == 0xFF && sign_a != sign_b) return F32_QNAN;
        return a;
    }
    if (exp_b == 0xFF) return b;

    if (exp_a != 0) sig_a |= 0x00800000u; else exp_a = 1;
    if (exp_b != 0) sig_b |= 0x00800000u; else exp_b = 1;

    sig_a <<= 8;
    sig_b <<= 8;

    int exp_diff = exp_a - exp_b;
    if (exp_diff > 0) {
        if (exp_diff < 32) {
            u32 lost = sig_b & ((1u << exp_diff) - 1);
            sig_b >>= exp_diff;
            if (lost) sig_b |= 1; // sticky
        } else { sig_b = (sig_b != 0) ? 1 : 0; }
    } else if (exp_diff < 0) {
        exp_diff = -exp_diff;
        if (exp_diff < 32) {
            u32 lost = sig_a & ((1u << exp_diff) - 1);
            sig_a >>= exp_diff;
            if (lost) sig_a |= 1; // sticky
        } else { sig_a = (sig_a != 0) ? 1 : 0; }
        exp_a = exp_b;
    }

    u32 result_sig;
    int result_sign;
    int result_exp = exp_a;

    if (sign_a == sign_b) {
        u64 wide_sum = (u64)sig_a + (u64)sig_b;
        result_sign = sign_a;
        if (wide_sum == 0) return result_sign ? F32_SIGN_BIT : 0;
        if (wide_sum >= (1ull << 32)) {
            // Carry: shift right by 1, preserving sticky bit
            result_sig = (u32)(wide_sum >> 1);
            if (wide_sum & 1) result_sig |= 1;
            result_exp++;
        } else {
            result_sig = (u32)wide_sum;
        }
    } else {
        if (sig_a >= sig_b) {
            result_sig = sig_a - sig_b;
            result_sign = sign_a;
        } else {
            result_sig = sig_b - sig_a;
            result_sign = sign_b;
        }
        if (result_sig == 0) return 0;
        while (result_sig < (0x00800000u << 8) && result_exp > 0) {
            result_sig <<= 1;
            result_exp--;
        }
    }

    return f32_round_pack(result_sign, result_exp, result_sig);
}

static __attribute__((always_inline)) u32 f32_eq_impl(u32 a, u32 b) {
    if (f32_is_nan(a) || f32_is_nan(b)) return 0;
    if ((a | b) << 1 == 0) return 1;
    return a == b ? 1 : 0;
}

static __attribute__((always_inline)) u32 f32_lt_impl(u32 a, u32 b) {
    if (f32_is_nan(a) || f32_is_nan(b)) return 0;
    int sa = f32_sign(a), sb = f32_sign(b);
    if (sa != sb) {
        if ((a | b) << 1 == 0) return 0;
        return sa ? 1 : 0;
    }
    if (a == b) return 0;
    return (a < b) ^ sa ? 1 : 0;
}

static __attribute__((always_inline)) u32 f32_le_impl(u32 a, u32 b) {
    if (f32_is_nan(a) || f32_is_nan(b)) return 0;
    int sa = f32_sign(a), sb = f32_sign(b);
    if (sa != sb) {
        if ((a | b) << 1 == 0) return 1;
        return sa ? 1 : 0;
    }
    if (a == b) return 1;
    return (a < b) ^ sa ? 1 : 0;
}

// ---- Exported functions ----

__attribute__((export_name("f32_add")))
u32 f32_add(u32 a, u32 b) { return f32_add_impl(a, b); }

__attribute__((export_name("f32_sub")))
u32 f32_sub(u32 a, u32 b) { return f32_add_impl(a, b ^ F32_SIGN_BIT); }

__attribute__((export_name("f32_mul")))
u32 f32_mul(u32 a, u32 b) {
    if (f32_is_nan(a)) return F32_QNAN;
    if (f32_is_nan(b)) return F32_QNAN;

    int sign = f32_sign(a) ^ f32_sign(b);
    int exp_a = f32_exp(a);
    int exp_b = f32_exp(b);
    u32 sig_a = f32_frac(a);
    u32 sig_b = f32_frac(b);

    if (exp_a == 0xFF) {
        if (exp_b == 0 && sig_b == 0) return F32_QNAN;
        return (sign ? F32_SIGN_BIT : 0) | F32_INF;
    }
    if (exp_b == 0xFF) {
        if (exp_a == 0 && sig_a == 0) return F32_QNAN;
        return (sign ? F32_SIGN_BIT : 0) | F32_INF;
    }

    if ((exp_a == 0 && sig_a == 0) || (exp_b == 0 && sig_b == 0)) {
        return sign ? F32_SIGN_BIT : 0;
    }

    if (exp_a != 0) { sig_a |= 0x00800000u; } else {
        // Subnormal: normalize using binary search (avoids i32.clz)
        exp_a = 1;
        int shift = 0;
        if (sig_a < 0x00000100u) { shift += 16; sig_a <<= 16; }
        if (sig_a < 0x00010000u) { shift += 8;  sig_a <<= 8; }
        if (sig_a < 0x00100000u) { shift += 4;  sig_a <<= 4; }
        if (sig_a < 0x00400000u) { shift += 2;  sig_a <<= 2; }
        if (sig_a < 0x00800000u) { shift += 1;  sig_a <<= 1; }
        exp_a -= shift;
    }
    if (exp_b != 0) { sig_b |= 0x00800000u; } else {
        // Subnormal: normalize using binary search (avoids i32.clz)
        exp_b = 1;
        int shift = 0;
        if (sig_b < 0x00000100u) { shift += 16; sig_b <<= 16; }
        if (sig_b < 0x00010000u) { shift += 8;  sig_b <<= 8; }
        if (sig_b < 0x00100000u) { shift += 4;  sig_b <<= 4; }
        if (sig_b < 0x00400000u) { shift += 2;  sig_b <<= 2; }
        if (sig_b < 0x00800000u) { shift += 1;  sig_b <<= 1; }
        exp_b -= shift;
    }

    u64 product = (u64)sig_a * (u64)sig_b;
    int result_exp = exp_a + exp_b - F32_EXP_BIAS;

    // product is at most 48 bits. Check normalization on u64 before extracting to u32.
    u32 result_sig;
    if (product & (1ull << 47)) {
        // Leading bit at position 47: extract [47:16] as sig+round, [15:0] as sticky
        u32 sticky = (u32)(product & 0xFFFF) ? 1 : 0;
        result_sig = (u32)(product >> 16);
        if (sticky) result_sig |= 1;
        result_exp++;
    } else {
        // Leading bit at position 46: extract [46:15] as sig+round, [14:0] as sticky
        u32 sticky = (u32)(product & 0x7FFF) ? 1 : 0;
        result_sig = (u32)(product >> 15);
        if (sticky) result_sig |= 1;
    }

    return f32_round_pack(sign, result_exp, result_sig);
}

__attribute__((export_name("f32_div")))
u32 f32_div(u32 a, u32 b) {
    if (f32_is_nan(a)) return F32_QNAN;
    if (f32_is_nan(b)) return F32_QNAN;

    int sign = f32_sign(a) ^ f32_sign(b);
    int exp_a = f32_exp(a);
    int exp_b = f32_exp(b);
    u32 sig_a = f32_frac(a);
    u32 sig_b = f32_frac(b);

    if (exp_a == 0xFF && exp_b == 0xFF) return F32_QNAN;
    if (exp_a == 0xFF) return (sign ? F32_SIGN_BIT : 0) | F32_INF;
    if (exp_b == 0xFF) return sign ? F32_SIGN_BIT : 0;
    if (exp_b == 0 && sig_b == 0) {
        if (exp_a == 0 && sig_a == 0) return F32_QNAN;
        return (sign ? F32_SIGN_BIT : 0) | F32_INF;
    }
    if (exp_a == 0 && sig_a == 0) return sign ? F32_SIGN_BIT : 0;

    if (exp_a != 0) { sig_a |= 0x00800000u; } else {
        exp_a = 1;
        int shift = 0;
        if (sig_a < 0x00000100u) { shift += 16; sig_a <<= 16; }
        if (sig_a < 0x00010000u) { shift += 8;  sig_a <<= 8; }
        if (sig_a < 0x00100000u) { shift += 4;  sig_a <<= 4; }
        if (sig_a < 0x00400000u) { shift += 2;  sig_a <<= 2; }
        if (sig_a < 0x00800000u) { shift += 1;  sig_a <<= 1; }
        exp_a -= shift;
    }
    if (exp_b != 0) { sig_b |= 0x00800000u; } else {
        exp_b = 1;
        int shift = 0;
        if (sig_b < 0x00000100u) { shift += 16; sig_b <<= 16; }
        if (sig_b < 0x00010000u) { shift += 8;  sig_b <<= 8; }
        if (sig_b < 0x00100000u) { shift += 4;  sig_b <<= 4; }
        if (sig_b < 0x00400000u) { shift += 2;  sig_b <<= 2; }
        if (sig_b < 0x00800000u) { shift += 1;  sig_b <<= 1; }
        exp_b -= shift;
    }

    int result_exp = exp_a - exp_b + F32_EXP_BIAS;

    // When sig_a < sig_b, quotient < 2^31 (implicit 1 at bit 30 not 31).
    // Use an extra shift bit and decrement exponent to compensate.
    u64 dividend;
    if (sig_a < sig_b) {
        dividend = (u64)sig_a << 32;
        result_exp--;
    } else {
        dividend = (u64)sig_a << 31;
    }
    u32 quotient = (u32)(dividend / sig_b);
    u32 remainder = (u32)(dividend % sig_b);
    if (remainder) quotient |= 1;

    return f32_round_pack(sign, result_exp, quotient);
}

// ---- Comparison ----

__attribute__((export_name("f32_eq")))
u32 f32_eq(u32 a, u32 b) { return f32_eq_impl(a, b); }

__attribute__((export_name("f32_ne")))
u32 f32_ne(u32 a, u32 b) { return f32_eq_impl(a, b) ? 0 : 1; }

__attribute__((export_name("f32_lt")))
u32 f32_lt(u32 a, u32 b) { return f32_lt_impl(a, b); }

__attribute__((export_name("f32_le")))
u32 f32_le(u32 a, u32 b) { return f32_le_impl(a, b); }

__attribute__((export_name("f32_gt")))
u32 f32_gt(u32 a, u32 b) { return f32_lt_impl(b, a); }

__attribute__((export_name("f32_ge")))
u32 f32_ge(u32 a, u32 b) { return f32_le_impl(b, a); }

// ---- Unary ----

__attribute__((export_name("f32_abs")))
u32 f32_abs(u32 a) { return a & ~F32_SIGN_BIT; }

__attribute__((export_name("f32_neg")))
u32 f32_neg(u32 a) { return a ^ F32_SIGN_BIT; }

__attribute__((export_name("f32_copysign")))
u32 f32_copysign(u32 a, u32 b) {
    return (a & ~F32_SIGN_BIT) | (b & F32_SIGN_BIT);
}

__attribute__((export_name("f32_ceil")))
u32 f32_ceil(u32 a) {
    if (f32_is_nan(a)) return F32_QNAN;
    if (f32_is_inf(a)) return a;
    int sign = f32_sign(a);
    int exp = f32_exp(a);
    if (exp == 0 && f32_frac(a) == 0) return a;

    int shift = F32_EXP_BIAS + F32_EXP_SHIFT - exp;
    if (shift <= 0) return a;
    if (shift > 23) {
        return sign ? F32_SIGN_BIT : 0x3F800000u;
    }
    u32 mask = (1u << shift) - 1;
    if (!sign && (a & mask)) {
        a += (1u << shift);
    }
    return a & ~mask;
}

__attribute__((export_name("f32_floor")))
u32 f32_floor(u32 a) {
    if (f32_is_nan(a)) return F32_QNAN;
    if (f32_is_inf(a)) return a;
    int sign = f32_sign(a);
    int exp = f32_exp(a);
    if (exp == 0 && f32_frac(a) == 0) return a;

    int shift = F32_EXP_BIAS + F32_EXP_SHIFT - exp;
    if (shift <= 0) return a;
    if (shift > 23) {
        return sign ? 0xBF800000u : 0;
    }
    u32 mask = (1u << shift) - 1;
    if (sign && (a & mask)) {
        a += (1u << shift);
    }
    return a & ~mask;
}

__attribute__((export_name("f32_trunc")))
u32 f32_trunc(u32 a) {
    if (f32_is_nan(a)) return F32_QNAN;
    if (f32_is_inf(a)) return a;
    int exp = f32_exp(a);
    if (exp == 0 && f32_frac(a) == 0) return a;

    int shift = F32_EXP_BIAS + F32_EXP_SHIFT - exp;
    if (shift <= 0) return a;
    if (shift > 23) {
        return a & F32_SIGN_BIT;
    }
    u32 mask = (1u << shift) - 1;
    return a & ~mask;
}

__attribute__((export_name("f32_nearest")))
u32 f32_nearest(u32 a) {
    if (f32_is_nan(a)) return F32_QNAN;
    if (f32_is_inf(a)) return a;
    int exp = f32_exp(a);
    if (exp == 0 && f32_frac(a) == 0) return a;

    int shift = F32_EXP_BIAS + F32_EXP_SHIFT - exp;
    if (shift <= 0) return a;
    if (shift > 24) {
        return a & F32_SIGN_BIT;
    }
    if (shift == 24) {
        return a & F32_SIGN_BIT;
    }

    u32 mask = (1u << shift) - 1;
    u32 half = 1u << (shift - 1);
    u32 frac_bits = a & mask;

    if (frac_bits > half || (frac_bits == half && ((a >> shift) & 1))) {
        a += (1u << shift);
    }
    return a & ~mask;
}

__attribute__((export_name("f32_sqrt")))
u32 f32_sqrt(u32 a) {
    if (f32_is_nan(a)) return F32_QNAN;
    if (a == 0 || a == F32_SIGN_BIT) return a;
    if (f32_sign(a)) return F32_QNAN;
    if (f32_is_inf(a)) return a;

    int exp = f32_exp(a);
    u32 sig = f32_frac(a);
    if (exp == 0) {
        // Binary search normalization to avoid i32.clz WASM instruction
        int shift = 0;
        if (sig < 0x00000100u) { shift += 16; sig <<= 16; }
        if (sig < 0x00010000u) { shift += 8;  sig <<= 8; }
        if (sig < 0x00100000u) { shift += 4;  sig <<= 4; }
        if (sig < 0x00400000u) { shift += 2;  sig <<= 2; }
        if (sig < 0x00800000u) { shift += 1;  sig <<= 1; }
        exp = 1 - shift;
    }
    sig |= 0x00800000u;

    exp -= F32_EXP_BIAS;
    if (exp & 1) { sig <<= 1; }
    exp >>= 1;
    exp += F32_EXP_BIAS;

    // Digit-by-digit square root algorithm.
    // sig is 24-25 bits. We compute 32 bits of sqrt from 64 bits of input.
    // Input: sig << 39 (padded to 64 bits). Output: 32-bit root (24 mantissa + 8 guard).
    u64 n = (u64)sig << 39;
    u64 rem = 0;
    u32 root = 0;
    for (int i = 31; i >= 0; i--) {
        rem = (rem << 2) | ((n >> (2 * i)) & 3);
        u64 trial = ((u64)root << 2) | 1;
        if (rem >= trial) {
            rem -= trial;
            root = (root << 1) | 1;
        } else {
            root <<= 1;
        }
    }
    u32 result_sig = root;
    if (rem > 0) result_sig |= 1;  // sticky bit for rounding
    return f32_round_pack(0, exp, result_sig);
}

// ---- Min / Max ----

__attribute__((export_name("f32_min")))
u32 f32_min(u32 a, u32 b) {
    if (f32_is_nan(a)) return F32_QNAN;
    if (f32_is_nan(b)) return F32_QNAN;
    if (f32_lt_impl(a, b)) return a;
    if (f32_lt_impl(b, a)) return b;
    return a | b;
}

__attribute__((export_name("f32_max")))
u32 f32_max(u32 a, u32 b) {
    if (f32_is_nan(a)) return F32_QNAN;
    if (f32_is_nan(b)) return F32_QNAN;
    if (f32_lt_impl(b, a)) return a;
    if (f32_lt_impl(a, b)) return b;
    return a & b;
}

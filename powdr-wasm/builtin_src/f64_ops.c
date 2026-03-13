// Software float64 operations for WOMIR.
// f64 values are passed as u64 (i64 in WASM), matching the WASM f64 stack type.

typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;
typedef long long i64;

_Static_assert(sizeof(u32) == 4, "u32 must be 4 bytes");
_Static_assert(sizeof(u64) == 8, "u64 must be 8 bytes");

#define F64_SIGN_BIT   0x8000000000000000ull
#define F64_EXP_MASK   0x7FF0000000000000ull
#define F64_FRAC_MASK  0x000FFFFFFFFFFFFFull
#define F64_EXP_BIAS   1023
#define F64_EXP_SHIFT  52
#define F64_QNAN       0x7FF8000000000000ull
#define F64_INF_VAL    0x7FF0000000000000ull

static inline int f64_is_nan(u64 a) {
    return (a & F64_EXP_MASK) == F64_EXP_MASK && (a & F64_FRAC_MASK) != 0;
}
static inline int f64_is_inf(u64 a) {
    return (a & ~F64_SIGN_BIT) == F64_EXP_MASK;
}
static inline int f64_sign(u64 a) { return (int)(a >> 63); }
static inline int f64_exp(u64 a) { return (int)((a >> F64_EXP_SHIFT) & 0x7FF); }
static inline u64 f64_frac(u64 a) { return a & F64_FRAC_MASK; }

static inline u64 f64_make(int sign, int exp, u64 frac) {
    return ((u64)sign << 63) | ((u64)(exp & 0x7FF) << 52) | (frac & F64_FRAC_MASK);
}

// 128-bit multiplication helper
static __attribute__((always_inline)) void mul64(u64 a, u64 b, u64 *rhi, u64 *rlo) {
    u32 a0 = (u32)a, a1 = (u32)(a >> 32);
    u32 b0 = (u32)b, b1 = (u32)(b >> 32);
    u64 p00 = (u64)a0 * b0;
    u64 p01 = (u64)a0 * b1;
    u64 p10 = (u64)a1 * b0;
    u64 p11 = (u64)a1 * b1;
    u64 mid = (p00 >> 32) + (u32)p01 + (u32)p10;
    *rlo = ((u32)p00) | (mid << 32);
    *rhi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// Round and pack: rsig has 8 extra bits for rounding
static __attribute__((always_inline)) u64 f64_round_pack(int sign, int rexp, u64 rsig) {
    if (rexp >= 0x7FF) return ((u64)sign << 63) | F64_INF_VAL;
    if (rexp <= 0) {
        // Subnormal: shift right to make exp=0 format
        int shift = 1 - rexp;
        if (shift > 63) return (u64)sign << 63;
        u64 sticky_bits = rsig & ((1ull << shift) - 1);
        rsig >>= shift;
        if (sticky_bits) rsig |= 1;
        rexp = 0;
    }

    u32 round = (u32)(rsig >> 7) & 1;
    u32 sticky = (u32)(rsig & 0x7F);
    rsig >>= 8;
    rsig += (round & (sticky | (rsig & 1)));
    if (rsig >= (1ull << 53)) { rsig >>= 1; rexp++; }

    if (rexp >= 0x7FF) return ((u64)sign << 63) | F64_INF_VAL;

    // If exp > 0 but no implicit bit set, result is subnormal
    if (rexp > 0 && rsig < (1ull << 52)) {
        rexp = 0;
    }

    return f64_make(sign, rexp, rsig);
}

// ---- Internal helpers (always inlined) ----

static __attribute__((always_inline)) u64 f64_add_impl(u64 a, u64 b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return F64_QNAN;

    int sa = f64_sign(a), sb = f64_sign(b);
    int ea = f64_exp(a), eb = f64_exp(b);
    u64 fa = f64_frac(a), fb = f64_frac(b);

    if (ea == 0x7FF) {
        if (eb == 0x7FF && sa != sb) return F64_QNAN;
        return a;
    }
    if (eb == 0x7FF) return b;

    if (ea != 0) fa |= (1ull << 52); else ea = 1;
    if (eb != 0) fb |= (1ull << 52); else eb = 1;

    fa <<= 8;
    fb <<= 8;

    int diff = ea - eb;
    if (diff > 0) {
        if (diff < 64) {
            u64 lost = fb & ((1ull << diff) - 1);
            fb >>= diff;
            if (lost) fb |= 1;
        } else { fb = (fb != 0) ? 1 : 0; }
    } else if (diff < 0) {
        diff = -diff;
        if (diff < 64) {
            u64 lost = fa & ((1ull << diff) - 1);
            fa >>= diff;
            if (lost) fa |= 1;
        } else { fa = (fa != 0) ? 1 : 0; }
        ea = eb;
    }

    u64 rsig; int rsign; int rexp = ea;

    if (sa == sb) {
        rsig = fa + fb;
        rsign = sa;
        if (rsig == 0) return rsign ? F64_SIGN_BIT : 0;
        if (rsig >= (2ull << 60)) { rsig >>= 1; rexp++; }
    } else {
        if (fa >= fb) { rsig = fa - fb; rsign = sa; }
        else { rsig = fb - fa; rsign = sb; }
        if (rsig == 0) return 0;
        while (rsig < (1ull << 60) && rexp > 0) { rsig <<= 1; rexp--; }
    }

    return f64_round_pack(rsign, rexp, rsig);
}

static __attribute__((always_inline)) u32 f64_eq_impl(u64 a, u64 b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return 0;
    if ((a | b) << 1 == 0) return 1;
    return a == b ? 1 : 0;
}

static __attribute__((always_inline)) u32 f64_lt_impl(u64 a, u64 b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return 0;
    int sa = f64_sign(a), sb = f64_sign(b);
    if (sa != sb) {
        if ((a | b) << 1 == 0) return 0;
        return sa ? 1 : 0;
    }
    if (a == b) return 0;
    return (a < b) ^ sa ? 1 : 0;
}

static __attribute__((always_inline)) u32 f64_le_impl(u64 a, u64 b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return 0;
    int sa = f64_sign(a), sb = f64_sign(b);
    if (sa != sb) {
        if ((a | b) << 1 == 0) return 1;
        return sa ? 1 : 0;
    }
    if (a == b) return 1;
    return (a < b) ^ sa ? 1 : 0;
}

// ---- Exported Arithmetic ----

__attribute__((export_name("f64_add")))
u64 f64_add(u64 a, u64 b) { return f64_add_impl(a, b); }

__attribute__((export_name("f64_sub")))
u64 f64_sub(u64 a, u64 b) { return f64_add_impl(a, b ^ F64_SIGN_BIT); }

__attribute__((export_name("f64_mul")))
u64 f64_mul(u64 a, u64 b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return F64_QNAN;

    int sign = f64_sign(a) ^ f64_sign(b);
    int ea = f64_exp(a), eb = f64_exp(b);
    u64 fa = f64_frac(a), fb = f64_frac(b);

    if (ea == 0x7FF) {
        if (eb == 0 && fb == 0) return F64_QNAN;
        return ((u64)sign << 63) | F64_INF_VAL;
    }
    if (eb == 0x7FF) {
        if (ea == 0 && fa == 0) return F64_QNAN;
        return ((u64)sign << 63) | F64_INF_VAL;
    }
    if ((ea == 0 && fa == 0) || (eb == 0 && fb == 0)) {
        return (u64)sign << 63;
    }

    if (ea != 0) { fa |= (1ull << 52); } else {
        ea = 1;
        int shift = 0;
        if (fa < 0x200000ull)         { shift += 32; fa <<= 32; }
        if (fa < 0x2000000000ull)     { shift += 16; fa <<= 16; }
        if (fa < 0x200000000000ull)   { shift += 8;  fa <<= 8; }
        if (fa < 0x2000000000000ull)  { shift += 4;  fa <<= 4; }
        if (fa < 0x8000000000000ull)  { shift += 2;  fa <<= 2; }
        if (fa < 0x10000000000000ull) { shift += 1;  fa <<= 1; }
        ea -= shift;
    }
    if (eb != 0) { fb |= (1ull << 52); } else {
        eb = 1;
        int shift = 0;
        if (fb < 0x200000ull)         { shift += 32; fb <<= 32; }
        if (fb < 0x2000000000ull)     { shift += 16; fb <<= 16; }
        if (fb < 0x200000000000ull)   { shift += 8;  fb <<= 8; }
        if (fb < 0x2000000000000ull)  { shift += 4;  fb <<= 4; }
        if (fb < 0x8000000000000ull)  { shift += 2;  fb <<= 2; }
        if (fb < 0x10000000000000ull) { shift += 1;  fb <<= 1; }
        eb -= shift;
    }

    int rexp = ea + eb - F64_EXP_BIAS;

    u64 phi, plo;
    mul64(fa, fb, &phi, &plo);

    // Product implicit 1 at bit 104. Need 60 bits for rounding. Shift right by 44.
    u64 rsig = (phi << 20) | (plo >> 44);
    u32 sticky = (u32)(plo & ((1ull << 44) - 1)) ? 1 : 0;
    if (sticky) rsig |= 1;

    if (rsig >= (2ull << 60)) { sticky = (u32)(rsig & 1); rsig >>= 1; if (sticky) rsig |= 1; rexp++; }

    return f64_round_pack(sign, rexp, rsig);
}

__attribute__((export_name("f64_div")))
u64 f64_div(u64 a, u64 b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return F64_QNAN;

    int sign = f64_sign(a) ^ f64_sign(b);
    int ea = f64_exp(a), eb = f64_exp(b);
    u64 fa = f64_frac(a), fb = f64_frac(b);

    if (ea == 0x7FF && eb == 0x7FF) return F64_QNAN;
    if (ea == 0x7FF) return ((u64)sign << 63) | F64_INF_VAL;
    if (eb == 0x7FF) return (u64)sign << 63;
    if (eb == 0 && fb == 0) {
        if (ea == 0 && fa == 0) return F64_QNAN;
        return ((u64)sign << 63) | F64_INF_VAL;
    }
    if (ea == 0 && fa == 0) return (u64)sign << 63;

    if (ea != 0) { fa |= (1ull << 52); } else {
        ea = 1;
        int shift = 0;
        if (fa < 0x200000ull)         { shift += 32; fa <<= 32; }
        if (fa < 0x2000000000ull)     { shift += 16; fa <<= 16; }
        if (fa < 0x200000000000ull)   { shift += 8;  fa <<= 8; }
        if (fa < 0x2000000000000ull)  { shift += 4;  fa <<= 4; }
        if (fa < 0x8000000000000ull)  { shift += 2;  fa <<= 2; }
        if (fa < 0x10000000000000ull) { shift += 1;  fa <<= 1; }
        ea -= shift;
    }
    if (eb != 0) { fb |= (1ull << 52); } else {
        eb = 1;
        int shift = 0;
        if (fb < 0x200000ull)         { shift += 32; fb <<= 32; }
        if (fb < 0x2000000000ull)     { shift += 16; fb <<= 16; }
        if (fb < 0x200000000000ull)   { shift += 8;  fb <<= 8; }
        if (fb < 0x2000000000000ull)  { shift += 4;  fb <<= 4; }
        if (fb < 0x8000000000000ull)  { shift += 2;  fb <<= 2; }
        if (fb < 0x10000000000000ull) { shift += 1;  fb <<= 1; }
        eb -= shift;
    }

    int rexp = ea - eb + F64_EXP_BIAS;

    // Ensure fa >= fb for full precision. If fa < fb, shift fa left.
    if (fa < fb) {
        fa <<= 1;
        rexp--;
    }

    // Shift-subtract division: fa >= fb, so integer bit is always 1.
    // Generate 61 bits: 1 integer + 60 fractional (53 mantissa + 7 guard + sticky).
    u64 q = 0;
    u64 rem = fa - fb;
    q = 1ull << 60;
    for (int i = 59; i >= 0; i--) {
        rem <<= 1;
        if (rem >= fb) {
            rem -= fb;
            q |= (1ull << i);
        }
    }
    if (rem) q |= 1;

    return f64_round_pack(sign, rexp, q);
}

// ---- Comparison (return u32/i32) ----

__attribute__((export_name("f64_eq")))
u32 f64_eq(u64 a, u64 b) { return f64_eq_impl(a, b); }

__attribute__((export_name("f64_ne")))
u32 f64_ne(u64 a, u64 b) { return f64_eq_impl(a, b) ? 0 : 1; }

__attribute__((export_name("f64_lt")))
u32 f64_lt(u64 a, u64 b) { return f64_lt_impl(a, b); }

__attribute__((export_name("f64_le")))
u32 f64_le(u64 a, u64 b) { return f64_le_impl(a, b); }

__attribute__((export_name("f64_gt")))
u32 f64_gt(u64 a, u64 b) { return f64_lt_impl(b, a); }

__attribute__((export_name("f64_ge")))
u32 f64_ge(u64 a, u64 b) { return f64_le_impl(b, a); }

// ---- Unary ----

__attribute__((export_name("f64_abs")))
u64 f64_abs(u64 a) { return a & ~F64_SIGN_BIT; }

__attribute__((export_name("f64_neg")))
u64 f64_neg(u64 a) { return a ^ F64_SIGN_BIT; }

__attribute__((export_name("f64_copysign")))
u64 f64_copysign(u64 a, u64 b) {
    return (a & ~F64_SIGN_BIT) | (b & F64_SIGN_BIT);
}

__attribute__((export_name("f64_ceil")))
u64 f64_ceil(u64 a) {
    if (f64_is_nan(a)) return F64_QNAN;
    if (f64_is_inf(a)) return a;
    int sign = f64_sign(a);
    int exp = f64_exp(a);
    if (exp == 0 && f64_frac(a) == 0) return a;

    int shift = F64_EXP_BIAS + F64_EXP_SHIFT - exp;
    if (shift <= 0) return a;
    if (shift > 52) {
        return sign ? F64_SIGN_BIT : 0x3FF0000000000000ull; // -0 or 1.0
    }
    u64 mask = (1ull << shift) - 1;
    if (!sign && (a & mask)) a += (1ull << shift);
    return a & ~mask;
}

__attribute__((export_name("f64_floor")))
u64 f64_floor(u64 a) {
    if (f64_is_nan(a)) return F64_QNAN;
    if (f64_is_inf(a)) return a;
    int sign = f64_sign(a);
    int exp = f64_exp(a);
    if (exp == 0 && f64_frac(a) == 0) return a;

    int shift = F64_EXP_BIAS + F64_EXP_SHIFT - exp;
    if (shift <= 0) return a;
    if (shift > 52) {
        return sign ? 0xBFF0000000000000ull : 0; // -1.0 or +0
    }
    u64 mask = (1ull << shift) - 1;
    if (sign && (a & mask)) a += (1ull << shift);
    return a & ~mask;
}

__attribute__((export_name("f64_trunc")))
u64 f64_trunc(u64 a) {
    if (f64_is_nan(a)) return F64_QNAN;
    if (f64_is_inf(a)) return a;
    int exp = f64_exp(a);
    if (exp == 0 && f64_frac(a) == 0) return a;

    int shift = F64_EXP_BIAS + F64_EXP_SHIFT - exp;
    if (shift <= 0) return a;
    if (shift > 52) return a & F64_SIGN_BIT;
    u64 mask = (1ull << shift) - 1;
    return a & ~mask;
}

__attribute__((export_name("f64_nearest")))
u64 f64_nearest(u64 a) {
    if (f64_is_nan(a)) return F64_QNAN;
    if (f64_is_inf(a)) return a;
    int exp = f64_exp(a);
    if (exp == 0 && f64_frac(a) == 0) return a;

    int shift = F64_EXP_BIAS + F64_EXP_SHIFT - exp;
    if (shift <= 0) return a;
    if (shift > 53) return a & F64_SIGN_BIT;
    if (shift == 53) return a & F64_SIGN_BIT;

    u64 mask = (1ull << shift) - 1;
    u64 half = 1ull << (shift - 1);
    u64 frac_bits = a & mask;

    if (frac_bits > half || (frac_bits == half && ((a >> shift) & 1))) {
        a += (1ull << shift);
    }
    return a & ~mask;
}

__attribute__((export_name("f64_sqrt")))
u64 f64_sqrt(u64 a) {
    if (f64_is_nan(a)) return F64_QNAN;
    if (a == 0 || a == F64_SIGN_BIT) return a;
    if (f64_sign(a)) return F64_QNAN;
    if (f64_is_inf(a)) return a;

    int exp = f64_exp(a);
    u64 sig = f64_frac(a);
    if (exp == 0) {
        // Binary search normalization to avoid i64.clz WASM instruction
        int shift = 0;
        if (sig < 0x200000ull)         { shift += 32; sig <<= 32; }
        if (sig < 0x2000000000ull)     { shift += 16; sig <<= 16; }
        if (sig < 0x200000000000ull)   { shift += 8;  sig <<= 8; }
        if (sig < 0x2000000000000ull)  { shift += 4;  sig <<= 4; }
        if (sig < 0x8000000000000ull)  { shift += 2;  sig <<= 2; }
        if (sig < 0x10000000000000ull) { shift += 1;  sig <<= 1; }
        exp = 1 - shift;
    }
    sig |= (1ull << 52);

    exp -= F64_EXP_BIAS;
    if (exp & 1) sig <<= 1;
    exp >>= 1;
    exp += F64_EXP_BIAS;

    // Digit-by-digit square root algorithm.
    // sig is 53-54 bits. We compute 61 bits of sqrt (53 mantissa + 8 guard).
    // Input: sig << 68 (122 bits, split into two u64s). Output: 61-bit root.
    u64 n_hi = sig << 4;  // upper 64 bits of 122-bit padded input
    u64 rem = 0;
    u64 root = 0;
    // 29 iterations processing upper bits from n_hi
    for (int i = 60; i >= 32; i--) {
        rem = (rem << 2) | ((n_hi >> (2 * (i - 32))) & 3);
        u64 trial = (root << 2) | 1;
        if (rem >= trial) {
            rem -= trial;
            root = (root << 1) | 1;
        } else {
            root <<= 1;
        }
    }
    // 32 iterations processing lower bits (all zeros since n_lo = 0)
    for (int i = 31; i >= 0; i--) {
        rem <<= 2;
        u64 trial = (root << 2) | 1;
        if (rem >= trial) {
            rem -= trial;
            root = (root << 1) | 1;
        } else {
            root <<= 1;
        }
    }
    if (rem > 0) root |= 1;  // sticky bit for rounding
    return f64_round_pack(0, exp, root);
}

// ---- Min / Max ----

__attribute__((export_name("f64_min")))
u64 f64_min(u64 a, u64 b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return F64_QNAN;
    if (f64_lt_impl(a, b)) return a;
    if (f64_lt_impl(b, a)) return b;
    return a | b;
}

__attribute__((export_name("f64_max")))
u64 f64_max(u64 a, u64 b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return F64_QNAN;
    if (f64_lt_impl(b, a)) return a;
    if (f64_lt_impl(a, b)) return b;
    return a & b;
}

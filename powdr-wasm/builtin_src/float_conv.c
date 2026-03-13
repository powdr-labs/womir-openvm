// Float conversion operations for WOMIR.
// Uses u64 directly for i64/f64 values to match WASM stack types.

typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;
typedef long long i64;

_Static_assert(sizeof(u32) == 4, "u32 must be 4 bytes");
_Static_assert(sizeof(u64) == 8, "u64 must be 8 bytes");

// ---- i32 -> f32 conversions ----

// Helper: round-to-nearest-even and pack f32 from (sign, exp, sig).
// sig has the implicit 1 at bit 24, with 1 guard bit at bit 0.
// `sticky` indicates bits beyond the guard bit were non-zero.
static u32 f32_conv_round_pack(int sign, int exp, u32 sig, u32 sticky) {
    u32 guard = sig & 1;
    sig >>= 1;
    // Round to nearest even: round up if guard=1 and (sticky or lsb)
    sig += (guard & (sticky | (sig & 1)));
    if (sig >= (1u << 24)) { sig >>= 1; exp++; }
    return (sign ? 0x80000000u : 0) | ((u32)exp << 23) | (sig & 0x7FFFFF);
}

__attribute__((export_name("f32_convert_i32_s")))
u32 f32_convert_i32_s(i32 a) {
    if (a == 0) return 0;
    int sign = 0;
    u32 abs_val;
    if (a < 0) { sign = 1; abs_val = (u32)0 - (u32)a; }
    else { abs_val = (u32)a; }

    // Normalize to 25 bits (1 implicit + 23 fraction + 1 guard)
    int exp = 127 + 24;
    u32 sticky = 0;
    while (abs_val >= (1u << 25)) {
        sticky |= (abs_val & 1);
        abs_val >>= 1;
        exp++;
    }
    while (abs_val < (1u << 24)) { abs_val <<= 1; exp--; }

    return f32_conv_round_pack(sign, exp, abs_val, sticky);
}

__attribute__((export_name("f32_convert_i32_u")))
u32 f32_convert_i32_u(u32 a) {
    if (a == 0) return 0;

    int exp = 127 + 24;
    u32 sig = a;
    u32 sticky = 0;
    while (sig >= (1u << 25)) {
        sticky |= (sig & 1);
        sig >>= 1;
        exp++;
    }
    while (sig < (1u << 24)) { sig <<= 1; exp--; }

    return f32_conv_round_pack(0, exp, sig, sticky);
}

// ---- i64 -> f32 conversions ----

__attribute__((export_name("f32_convert_i64_s")))
u32 f32_convert_i64_s(i64 a) {
    if (a == 0) return 0;
    int sign = 0;
    u64 abs_val;
    if (a < 0) { sign = 1; abs_val = (u64)0 - (u64)a; }
    else { abs_val = (u64)a; }

    int exp = 127 + 24;
    u32 sticky = 0;
    while (abs_val >= (1u << 25)) {
        sticky |= (u32)(abs_val & 1);
        abs_val >>= 1;
        exp++;
    }
    while (abs_val < (1u << 24)) { abs_val <<= 1; exp--; }

    return f32_conv_round_pack(sign, exp, (u32)abs_val, sticky);
}

__attribute__((export_name("f32_convert_i64_u")))
u32 f32_convert_i64_u(u64 a) {
    if (a == 0) return 0;

    int exp = 127 + 24;
    u32 sticky = 0;
    while (a >= (1u << 25)) {
        sticky |= (u32)(a & 1);
        a >>= 1;
        exp++;
    }
    while (a < (1u << 24)) { a <<= 1; exp--; }

    return f32_conv_round_pack(0, exp, (u32)a, sticky);
}

// ---- i32 -> f64 conversions ----

__attribute__((export_name("f64_convert_i32_s")))
u64 f64_convert_i32_s(i32 a) {
    if (a == 0) return 0;
    int sign = 0;
    u64 abs_val;
    if (a < 0) { sign = 1; abs_val = (u64)((u32)0 - (u32)a); }
    else { abs_val = (u64)(u32)a; }

    int exp = 1023 + 52;
    while (abs_val >= (1ull << 53)) { abs_val >>= 1; exp++; }
    while (abs_val < (1ull << 52)) { abs_val <<= 1; exp--; }

    return ((u64)sign << 63) | ((u64)exp << 52) | (abs_val & 0xFFFFFFFFFFFFFull);
}

__attribute__((export_name("f64_convert_i32_u")))
u64 f64_convert_i32_u(u32 a) {
    if (a == 0) return 0;

    u64 sig = (u64)a;
    int exp = 1023 + 52;
    while (sig < (1ull << 52)) { sig <<= 1; exp--; }

    return ((u64)exp << 52) | (sig & 0xFFFFFFFFFFFFFull);
}

// ---- i64 -> f64 conversions ----

// Helper: round-to-nearest-even and pack f64 from (sign, exp, sig).
// sig has the implicit 1 at bit 53, with 1 guard bit at bit 0.
static u64 f64_conv_round_pack(int sign, int exp, u64 sig, u32 sticky) {
    u32 guard = (u32)(sig & 1);
    sig >>= 1;
    sig += (guard & (sticky | (u32)(sig & 1)));
    if (sig >= (1ull << 53)) { sig >>= 1; exp++; }
    return ((u64)sign << 63) | ((u64)exp << 52) | (sig & 0xFFFFFFFFFFFFFull);
}

__attribute__((export_name("f64_convert_i64_s")))
u64 f64_convert_i64_s(i64 a) {
    if (a == 0) return 0;
    int sign = 0;
    u64 abs_val;
    if (a < 0) { sign = 1; abs_val = (u64)0 - (u64)a; }
    else { abs_val = (u64)a; }

    int exp = 1023 + 53;
    u32 sticky = 0;
    while (abs_val >= (1ull << 54)) {
        sticky |= (u32)(abs_val & 1);
        abs_val >>= 1;
        exp++;
    }
    while (abs_val < (1ull << 53)) { abs_val <<= 1; exp--; }

    return f64_conv_round_pack(sign, exp, abs_val, sticky);
}

__attribute__((export_name("f64_convert_i64_u")))
u64 f64_convert_i64_u(u64 a) {
    if (a == 0) return 0;

    int exp = 1023 + 53;
    u32 sticky = 0;
    while (a >= (1ull << 54)) {
        sticky |= (u32)(a & 1);
        a >>= 1;
        exp++;
    }
    while (a < (1ull << 53)) { a <<= 1; exp--; }

    return f64_conv_round_pack(0, exp, a, sticky);
}

// ---- f32 -> i32 truncations ----

__attribute__((export_name("i32_trunc_f32_s")))
i32 i32_trunc_f32_s(u32 a) {
    int sign = (a >> 31) & 1;
    int exp = (int)((a >> 23) & 0xFF);
    u32 frac = a & 0x7FFFFF;

    if (exp == 0xFF) return 0;
    if (exp == 0) return 0;

    frac |= 0x800000;
    int shift = exp - 127 - 23;
    u32 result;
    if (shift >= 0) {
        if (shift > 7) return sign ? (i32)0x80000000 : 0x7FFFFFFF;
        result = frac << shift;
    } else {
        if (-shift >= 24) return 0;
        result = frac >> (-shift);
    }
    return sign ? -(i32)result : (i32)result;
}

__attribute__((export_name("i32_trunc_f32_u")))
u32 i32_trunc_f32_u(u32 a) {
    int sign = (a >> 31) & 1;
    if (sign) return 0;

    int exp = (int)((a >> 23) & 0xFF);
    u32 frac = a & 0x7FFFFF;

    if (exp == 0xFF) return 0;
    if (exp == 0) return 0;

    frac |= 0x800000;
    int shift = exp - 127 - 23;
    if (shift >= 0) {
        if (shift > 8) return 0xFFFFFFFF;
        return frac << shift;
    } else {
        if (-shift >= 24) return 0;
        return frac >> (-shift);
    }
}

// ---- f64 -> i32 truncations ----

__attribute__((export_name("i32_trunc_f64_s")))
i32 i32_trunc_f64_s(u64 a) {
    int sign = (int)(a >> 63);
    int exp = (int)((a >> 52) & 0x7FF);
    u64 frac = a & 0xFFFFFFFFFFFFFull;

    if (exp == 0x7FF) return 0;
    if (exp == 0) return 0;

    frac |= (1ull << 52);
    int shift = exp - 1023 - 52;
    i64 result;
    if (shift >= 0) {
        if (shift > 11) return sign ? (i32)0x80000000 : 0x7FFFFFFF;
        result = (i64)(frac << shift);
    } else {
        if (-shift >= 53) return 0;
        result = (i64)(frac >> (-shift));
    }
    return sign ? -(i32)result : (i32)result;
}

__attribute__((export_name("i32_trunc_f64_u")))
u32 i32_trunc_f64_u(u64 a) {
    int sign = (int)(a >> 63);
    if (sign) return 0;

    int exp = (int)((a >> 52) & 0x7FF);
    u64 frac = a & 0xFFFFFFFFFFFFFull;

    if (exp == 0x7FF) return 0;
    if (exp == 0) return 0;

    frac |= (1ull << 52);
    int shift = exp - 1023 - 52;
    if (shift >= 0) {
        if (shift > 12) return 0xFFFFFFFF;
        u64 result = frac << shift;
        if (result > 0xFFFFFFFF) return 0xFFFFFFFF;
        return (u32)result;
    } else {
        if (-shift >= 53) return 0;
        return (u32)(frac >> (-shift));
    }
}

// ---- f32 -> i64 truncations ----

__attribute__((export_name("i64_trunc_f32_s")))
i64 i64_trunc_f32_s(u32 a) {
    int sign = (a >> 31) & 1;
    int exp = (int)((a >> 23) & 0xFF);
    u32 frac = a & 0x7FFFFF;

    if (exp == 0xFF || exp == 0) return 0;

    frac |= 0x800000;
    int shift = exp - 127 - 23;
    u64 result;
    if (shift >= 0) {
        if (shift > 39) return sign ? (i64)0x8000000000000000ll : 0x7FFFFFFFFFFFFFFFll;
        result = (u64)frac << shift;
    } else {
        if (-shift >= 24) return 0;
        result = (u64)frac >> (-shift);
    }
    return sign ? -(i64)result : (i64)result;
}

__attribute__((export_name("i64_trunc_f32_u")))
u64 i64_trunc_f32_u(u32 a) {
    int sign = (a >> 31) & 1;
    if (sign) return 0;

    int exp = (int)((a >> 23) & 0xFF);
    u32 frac = a & 0x7FFFFF;

    if (exp == 0xFF || exp == 0) return 0;

    frac |= 0x800000;
    int shift = exp - 127 - 23;
    u64 result;
    if (shift >= 0) {
        if (shift > 40) return 0xFFFFFFFFFFFFFFFFull;
        result = (u64)frac << shift;
    } else {
        if (-shift >= 24) return 0;
        result = (u64)frac >> (-shift);
    }
    return result;
}

// ---- f64 -> i64 truncations ----

__attribute__((export_name("i64_trunc_f64_s")))
i64 i64_trunc_f64_s(u64 a) {
    int sign = (int)(a >> 63);
    int exp = (int)((a >> 52) & 0x7FF);
    u64 frac = a & 0xFFFFFFFFFFFFFull;

    if (exp == 0x7FF || exp == 0) return 0;

    frac |= (1ull << 52);
    int shift = exp - 1023 - 52;
    u64 result;
    if (shift >= 0) {
        if (shift > 11) return sign ? (i64)0x8000000000000000ll : 0x7FFFFFFFFFFFFFFFll;
        result = frac << shift;
    } else {
        if (-shift >= 53) return 0;
        result = frac >> (-shift);
    }
    return sign ? -(i64)result : (i64)result;
}

__attribute__((export_name("i64_trunc_f64_u")))
u64 i64_trunc_f64_u(u64 a) {
    int sign = (int)(a >> 63);
    if (sign) return 0;

    int exp = (int)((a >> 52) & 0x7FF);
    u64 frac = a & 0xFFFFFFFFFFFFFull;

    if (exp == 0x7FF || exp == 0) return 0;

    frac |= (1ull << 52);
    int shift = exp - 1023 - 52;
    u64 result;
    if (shift >= 0) {
        if (shift > 12) return 0xFFFFFFFFFFFFFFFFull;
        result = frac << shift;
    } else {
        if (-shift >= 53) return 0;
        result = frac >> (-shift);
    }
    return result;
}

// ---- saturating truncations ----
// Like trunc but clamp to integer range instead of trapping on overflow/NaN.

__attribute__((export_name("i32_trunc_sat_f32_s")))
i32 i32_trunc_sat_f32_s(u32 a) {
    int sign = (a >> 31) & 1;
    int exp = (int)((a >> 23) & 0xFF);
    u32 frac = a & 0x7FFFFF;

    if (exp == 0xFF) {
        if (frac) return 0; // NaN -> 0
        return sign ? (i32)0x80000000 : 0x7FFFFFFF; // +-Inf -> MIN/MAX
    }
    if (exp == 0) return 0;

    frac |= 0x800000;
    int shift = exp - 127 - 23;
    u32 result;
    if (shift >= 0) {
        if (shift > 7) return sign ? (i32)0x80000000 : 0x7FFFFFFF;
        result = frac << shift;
    } else {
        if (-shift >= 24) return 0;
        result = frac >> (-shift);
    }
    return sign ? -(i32)result : (i32)result;
}

__attribute__((export_name("i32_trunc_sat_f32_u")))
u32 i32_trunc_sat_f32_u(u32 a) {
    int sign = (a >> 31) & 1;
    if (sign) return 0; // negative -> 0

    int exp = (int)((a >> 23) & 0xFF);
    u32 frac = a & 0x7FFFFF;

    if (exp == 0xFF) {
        if (frac) return 0; // NaN -> 0
        return 0xFFFFFFFF; // +Inf -> MAX
    }
    if (exp == 0) return 0;

    frac |= 0x800000;
    int shift = exp - 127 - 23;
    if (shift >= 0) {
        if (shift > 8) return 0xFFFFFFFF;
        return frac << shift;
    } else {
        if (-shift >= 24) return 0;
        return frac >> (-shift);
    }
}

__attribute__((export_name("i32_trunc_sat_f64_s")))
i32 i32_trunc_sat_f64_s(u64 a) {
    int sign = (int)(a >> 63);
    int exp = (int)((a >> 52) & 0x7FF);
    u64 frac = a & 0xFFFFFFFFFFFFFull;

    if (exp == 0x7FF) {
        if (frac) return 0; // NaN -> 0
        return sign ? (i32)0x80000000 : 0x7FFFFFFF; // +-Inf -> MIN/MAX
    }
    if (exp == 0) return 0;

    frac |= (1ull << 52);
    int shift = exp - 1023 - 52;
    i64 result;
    if (shift >= 0) {
        if (shift > 11) return sign ? (i32)0x80000000 : 0x7FFFFFFF;
        result = (i64)(frac << shift);
    } else {
        if (-shift >= 53) return 0;
        result = (i64)(frac >> (-shift));
    }
    if (sign) {
        if (result > 0x80000000ull) return (i32)0x80000000;
        return -(i32)result;
    } else {
        if (result > 0x7FFFFFFF) return 0x7FFFFFFF;
        return (i32)result;
    }
}

__attribute__((export_name("i32_trunc_sat_f64_u")))
u32 i32_trunc_sat_f64_u(u64 a) {
    int sign = (int)(a >> 63);
    if (sign) return 0; // negative -> 0

    int exp = (int)((a >> 52) & 0x7FF);
    u64 frac = a & 0xFFFFFFFFFFFFFull;

    if (exp == 0x7FF) {
        if (frac) return 0; // NaN -> 0
        return 0xFFFFFFFF; // +Inf -> MAX
    }
    if (exp == 0) return 0;

    frac |= (1ull << 52);
    int shift = exp - 1023 - 52;
    if (shift >= 0) {
        if (shift > 12) return 0xFFFFFFFF;
        u64 result = frac << shift;
        if (result > 0xFFFFFFFF) return 0xFFFFFFFF;
        return (u32)result;
    } else {
        if (-shift >= 53) return 0;
        u64 result = frac >> (-shift);
        if (result > 0xFFFFFFFF) return 0xFFFFFFFF;
        return (u32)result;
    }
}

__attribute__((export_name("i64_trunc_sat_f32_s")))
i64 i64_trunc_sat_f32_s(u32 a) {
    int sign = (a >> 31) & 1;
    int exp = (int)((a >> 23) & 0xFF);
    u32 frac = a & 0x7FFFFF;

    if (exp == 0xFF) {
        if (frac) return 0; // NaN -> 0
        return sign ? (i64)0x8000000000000000ll : 0x7FFFFFFFFFFFFFFFll;
    }
    if (exp == 0) return 0;

    frac |= 0x800000;
    int shift = exp - 127 - 23;
    u64 result;
    if (shift >= 0) {
        if (shift > 39) return sign ? (i64)0x8000000000000000ll : 0x7FFFFFFFFFFFFFFFll;
        result = (u64)frac << shift;
    } else {
        if (-shift >= 24) return 0;
        result = (u64)frac >> (-shift);
    }
    return sign ? -(i64)result : (i64)result;
}

__attribute__((export_name("i64_trunc_sat_f32_u")))
u64 i64_trunc_sat_f32_u(u32 a) {
    int sign = (a >> 31) & 1;
    if (sign) return 0; // negative -> 0

    int exp = (int)((a >> 23) & 0xFF);
    u32 frac = a & 0x7FFFFF;

    if (exp == 0xFF) {
        if (frac) return 0; // NaN -> 0
        return 0xFFFFFFFFFFFFFFFFull; // +Inf -> MAX
    }
    if (exp == 0) return 0;

    frac |= 0x800000;
    int shift = exp - 127 - 23;
    u64 result;
    if (shift >= 0) {
        if (shift > 40) return 0xFFFFFFFFFFFFFFFFull;
        result = (u64)frac << shift;
    } else {
        if (-shift >= 24) return 0;
        result = (u64)frac >> (-shift);
    }
    return result;
}

__attribute__((export_name("i64_trunc_sat_f64_s")))
i64 i64_trunc_sat_f64_s(u64 a) {
    int sign = (int)(a >> 63);
    int exp = (int)((a >> 52) & 0x7FF);
    u64 frac = a & 0xFFFFFFFFFFFFFull;

    if (exp == 0x7FF) {
        if (frac) return 0; // NaN -> 0
        return sign ? (i64)0x8000000000000000ll : 0x7FFFFFFFFFFFFFFFll;
    }
    if (exp == 0) return 0;

    frac |= (1ull << 52);
    int shift = exp - 1023 - 52;
    u64 result;
    if (shift >= 0) {
        if (shift > 11) return sign ? (i64)0x8000000000000000ll : 0x7FFFFFFFFFFFFFFFll;
        result = frac << shift;
    } else {
        if (-shift >= 53) return 0;
        result = frac >> (-shift);
    }
    if (sign) {
        if (result > 0x8000000000000000ull) return (i64)0x8000000000000000ll;
        return -(i64)result;
    } else {
        if (result > 0x7FFFFFFFFFFFFFFFull) return 0x7FFFFFFFFFFFFFFFll;
        return (i64)result;
    }
}

__attribute__((export_name("i64_trunc_sat_f64_u")))
u64 i64_trunc_sat_f64_u(u64 a) {
    int sign = (int)(a >> 63);
    if (sign) return 0; // negative -> 0

    int exp = (int)((a >> 52) & 0x7FF);
    u64 frac = a & 0xFFFFFFFFFFFFFull;

    if (exp == 0x7FF) {
        if (frac) return 0; // NaN -> 0
        return 0xFFFFFFFFFFFFFFFFull; // +Inf -> MAX
    }
    if (exp == 0) return 0;

    frac |= (1ull << 52);
    int shift = exp - 1023 - 52;
    u64 result;
    if (shift >= 0) {
        if (shift > 11) return 0xFFFFFFFFFFFFFFFFull;
        result = frac << shift;
    } else {
        if (-shift >= 53) return 0;
        result = frac >> (-shift);
    }
    return result;
}

// ---- f32 <-> f64 conversions ----

__attribute__((export_name("f64_promote_f32")))
u64 f64_promote_f32(u32 a) {
    int sign = (a >> 31) & 1;
    int exp = (int)((a >> 23) & 0xFF);
    u32 frac = a & 0x7FFFFF;

    if (exp == 0xFF) {
        if (frac) return 0x7FF8000000000000ull; // qNaN
        return ((u64)sign << 63) | 0x7FF0000000000000ull; // Inf
    }
    if (exp == 0 && frac == 0) {
        return (u64)sign << 63; // +-0
    }
    if (exp == 0) {
        // Subnormal f32 -> normalize for f64
        // Binary search normalization to avoid i32.clz WASM instruction
        int shift = 0;
        if (frac < 0x00000100u) { shift += 16; frac <<= 16; }
        if (frac < 0x00010000u) { shift += 8;  frac <<= 8; }
        if (frac < 0x00100000u) { shift += 4;  frac <<= 4; }
        if (frac < 0x00400000u) { shift += 2;  frac <<= 2; }
        if (frac < 0x00800000u) { shift += 1;  frac <<= 1; }
        exp = 1 - shift;
        frac &= 0x7FFFFF;
    }
    int f64_exp = exp - 127 + 1023;
    return ((u64)sign << 63) | ((u64)f64_exp << 52) | ((u64)frac << 29);
}

__attribute__((export_name("f32_demote_f64")))
u32 f32_demote_f64(u64 a) {
    int sign = (int)(a >> 63);
    int exp = (int)((a >> 52) & 0x7FF);
    u64 frac = a & 0xFFFFFFFFFFFFFull;

    if (exp == 0x7FF) {
        if (frac) return 0x7FC00000u; // qNaN
        return (sign ? 0x80000000u : 0) | 0x7F800000u; // Inf
    }
    if (exp == 0 && frac == 0) {
        return sign ? 0x80000000u : 0; // +-0
    }

    // f64 subnormals: all smaller than f32 subnormal range, demote to +-0
    // (largest f64 subnormal ~ 2^-1022, smallest f32 subnormal = 2^-149)
    if (exp == 0) return sign ? 0x80000000u : 0;

    int f32_exp = exp - 1023 + 127;
    u64 sig = frac | (1ull << 52); // 53-bit significand with implicit bit

    // Round 53-bit sig down to 24-bit f32 significand.
    // Normal shift = 29 (53 - 24). For subnormals, shift more.
    int shift;
    if (f32_exp > 0) {
        shift = 29;
    } else {
        shift = 29 + (1 - f32_exp);
        f32_exp = 0;
        if (shift > 53) return sign ? 0x80000000u : 0;
    }

    u32 f32_sig = (u32)(sig >> shift);
    u32 guard = (u32)((sig >> (shift - 1)) & 1);
    u32 sticky = (sig & ((1ull << (shift - 1)) - 1)) ? 1 : 0;

    // Round to nearest even
    f32_sig += (guard & (sticky | (f32_sig & 1)));

    // Handle carry from rounding
    if (f32_exp == 0 && f32_sig >= 0x800000u) {
        // Subnormal rounded up to smallest normal
        f32_exp = 1;
    } else if (f32_sig >= (1u << 24)) {
        f32_sig >>= 1;
        f32_exp++;
    }

    if (f32_exp >= 0xFF) return (sign ? 0x80000000u : 0) | 0x7F800000u;

    return (sign ? 0x80000000u : 0) | ((u32)f32_exp << 23) | (f32_sig & 0x7FFFFF);
}

// WebAssembly float rules on raw IEEE-754 bit patterns.
// The f64 arithmetic primitives above come from the vendored metal-softfloat source.
static bool wasm_f32_nan(uint x) { return (x & 0x7fffffffu) > 0x7f800000u; }
static bool wasm_f64_nan(ulong x) { return (x & 0x7ffffffffffffffful) > 0x7ff0000000000000ul; }

static uint wasm_f32_min(uint a, uint b) {
    if (wasm_f32_nan(a) || wasm_f32_nan(b)) return 0x7fc00000u;
    if ((a & 0x7fffffffu) == 0u && (b & 0x7fffffffu) == 0u) return a | b;
    return __softfloat64_flt(__softfloat64_cvt_f32_to_f64(a), __softfloat64_cvt_f32_to_f64(b)) ? a : b;
}
static uint wasm_f32_max(uint a, uint b) {
    if (wasm_f32_nan(a) || wasm_f32_nan(b)) return 0x7fc00000u;
    if ((a & 0x7fffffffu) == 0u && (b & 0x7fffffffu) == 0u) return a & b;
    return __softfloat64_fgt(__softfloat64_cvt_f32_to_f64(a), __softfloat64_cvt_f32_to_f64(b)) ? a : b;
}
static ulong wasm_f64_min(ulong a, ulong b) {
    if (wasm_f64_nan(a) || wasm_f64_nan(b)) return 0x7ff8000000000000ul;
    if ((a & 0x7ffffffffffffffful) == 0ul && (b & 0x7ffffffffffffffful) == 0ul) return a | b;
    return __softfloat64_flt(a, b) ? a : b;
}
static ulong wasm_f64_max(ulong a, ulong b) {
    if (wasm_f64_nan(a) || wasm_f64_nan(b)) return 0x7ff8000000000000ul;
    if ((a & 0x7ffffffffffffffful) == 0ul && (b & 0x7ffffffffffffffful) == 0ul) return a & b;
    return __softfloat64_fgt(a, b) ? a : b;
}

static ulong wasm_f64_trunc(ulong a) {
    if (wasm_f64_nan(a)) return a | 0x0008000000000000ul;
    uint exponent = uint((a >> 52u) & 2047ul);
    if (exponent < 1023u) return a & 0x8000000000000000ul;
    if (exponent >= 1075u) return a;
    ulong mask = (1ul << (1075u - exponent)) - 1ul;
    return a & ~mask;
}
static ulong wasm_f64_floor(ulong a) {
    if (wasm_f64_nan(a)) return a | 0x0008000000000000ul;
    if ((a & 0x7ffffffffffffffful) == 0x7ff0000000000000ul) return a;
    ulong t = wasm_f64_trunc(a);
    return t != a && (a >> 63u) != 0ul ? __softfloat64_fsub(t, 0x3ff0000000000000ul, 0u) : t;
}
static ulong wasm_f64_ceil(ulong a) {
    if (wasm_f64_nan(a)) return a | 0x0008000000000000ul;
    if ((a & 0x7ffffffffffffffful) == 0x7ff0000000000000ul) return a;
    ulong t = wasm_f64_trunc(a);
    return t != a && (a >> 63u) == 0ul ? __softfloat64_fadd(t, 0x3ff0000000000000ul, 0u) : t;
}
static ulong wasm_f64_nearest(ulong a) {
    if (wasm_f64_nan(a)) return a | 0x0008000000000000ul;
    if ((a & 0x7ffffffffffffffful) == 0x7ff0000000000000ul) return a;
    ulong t = wasm_f64_trunc(a);
    if (t == a) return a;
    ulong fraction = __softfloat64_fsub(a, t, 0u) & 0x7ffffffffffffffful;
    bool more = __softfloat64_fgt(fraction, 0x3fe0000000000000ul);
    bool tie = __softfloat64_feq(fraction, 0x3fe0000000000000ul);
    uint exponent = uint((t >> 52u) & 2047ul);
    bool odd = exponent >= 1023u && exponent < 1075u && ((t >> (1075u - exponent)) & 1ul) != 0ul;
    if (!more && !(tie && odd)) return t;
    return (a >> 63u) != 0ul ? __softfloat64_fsub(t, 0x3ff0000000000000ul, 0u)
                            : __softfloat64_fadd(t, 0x3ff0000000000000ul, 0u);
}

static uint wasm_f32_add(uint a, uint b) {
    return __softfloat64_cvt_f64_to_f32(__softfloat64_fadd(__softfloat64_cvt_f32_to_f64(a), __softfloat64_cvt_f32_to_f64(b), 0u), 0u);
}
static uint wasm_f32_sub(uint a, uint b) {
    return __softfloat64_cvt_f64_to_f32(__softfloat64_fsub(__softfloat64_cvt_f32_to_f64(a), __softfloat64_cvt_f32_to_f64(b), 0u), 0u);
}
static uint wasm_f32_mul(uint a, uint b) {
    return __softfloat64_cvt_f64_to_f32(__softfloat64_fmul(__softfloat64_cvt_f32_to_f64(a), __softfloat64_cvt_f32_to_f64(b), 0u), 0u);
}
static uint wasm_f32_div(uint a, uint b) {
    return __softfloat64_cvt_f64_to_f32(__softfloat64_fdiv(__softfloat64_cvt_f32_to_f64(a), __softfloat64_cvt_f32_to_f64(b), 0u), 0u);
}
static uint wasm_f32_sqrt(uint a) {
    return __softfloat64_cvt_f64_to_f32(__softfloat64_fsqrt(__softfloat64_cvt_f32_to_f64(a), 0u), 0u);
}

// Convert a binary64 value to an integer. Set trap for NaN and out-of-range values.
// A saturating conversion returns the nearest endpoint instead.
static ulong wasm_f64_to_int(ulong a, uint width, bool signed_result, bool saturating, thread uint& trap) {
    bool negative = (a >> 63u) != 0ul;
    uint exponent = uint((a >> 52u) & 2047ul);
    ulong mantissa = a & 0x000ffffffffffffful;
    if (exponent == 2047u) {
        if (mantissa != 0ul) {
            if (!saturating) trap = 5u;
            return 0ul;
        }
        if (!saturating) trap = 2u;
        return negative ? (signed_result ? (width == 32u ? 0x80000000ul : 0x8000000000000000ul) : 0ul)
                        : (width == 32u ? (signed_result ? 0x7ffffffful : 0xfffffffful)
                                        : (signed_result ? 0x7ffffffffffffffful : 0xfffffffffffffffful));
    }
    int shift = int(exponent) - 1023;
    ulong limit = signed_result ? (negative ? (1ul << (width - 1u)) : ((1ul << (width - 1u)) - 1ul))
                                : (width == 32u ? 0xfffffffful : 0xfffffffffffffffful);
    if (shift >= int(width) || (negative && !signed_result && shift >= 0)) {
        if (!saturating) trap = 2u;
        return negative ? (signed_result ? limit : 0ul) : limit;
    }
    if (shift < 0) return 0ul;
    ulong magnitude = (1ul << 52u) | mantissa;
    magnitude = shift >= 52 ? magnitude << uint(shift - 52) : magnitude >> uint(52 - shift);
    if (magnitude > limit) {
        if (!saturating) trap = 2u;
        return negative ? (signed_result ? limit : 0ul) : limit;
    }
    ulong result = negative ? 0ul - magnitude : magnitude;
    return width == 32u ? ulong(uint(result)) : result;
}

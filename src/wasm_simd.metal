// A Wasm vector is held by one Metal thread as two little-endian 64-bit words.
inline ulong wasm_simd_mask(uint bits) {
    return bits == 64u ? ~0ul : ((1ul << bits) - 1ul);
}

inline ulong wasm_simd_lane(ulong2 value, uint bits, uint lane) {
    uint offset = lane * bits;
    return ((offset < 64u ? value.x : value.y) >> (offset & 63u)) & wasm_simd_mask(bits);
}

inline ulong2 wasm_simd_replace(ulong2 value, uint bits, uint lane, ulong part) {
    uint offset = lane * bits;
    ulong mask = wasm_simd_mask(bits);
    ulong shift = offset & 63u;
    if (offset < 64u) value.x = (value.x & ~(mask << shift)) | ((part & mask) << shift);
    else value.y = (value.y & ~(mask << shift)) | ((part & mask) << shift);
    return value;
}

inline long wasm_simd_signed(ulong value, uint bits) {
    ulong sign = 1ul << (bits - 1u);
    return as_type<long>((value ^ sign) - sign);
}

inline ulong2 wasm_simd_load(device const uchar* memory, ulong offset) {
    ulong2 value = ulong2(0ul);
    for (uint i = 0u; i < 8u; ++i) {
        value.x |= ulong(memory[offset + i]) << (i * 8u);
        value.y |= ulong(memory[offset + i + 8u]) << (i * 8u);
    }
    return value;
}

inline void wasm_simd_store(device uchar* memory, ulong offset, ulong2 value) {
    for (uint i = 0u; i < 8u; ++i) {
        memory[offset + i] = uchar(value.x >> (i * 8u));
        memory[offset + i + 8u] = uchar(value.y >> (i * 8u));
    }
}

inline ulong wasm_simd_read(device const uchar* memory, ulong offset, uint bytes) {
    ulong result = 0ul;
    for (uint i = 0u; i < bytes; ++i) result |= ulong(memory[offset + i]) << (i * 8u);
    return result;
}

inline void wasm_simd_write(device uchar* memory, ulong offset, uint bytes, ulong value) {
    for (uint i = 0u; i < bytes; ++i) memory[offset + i] = uchar(value >> (i * 8u));
}

inline ulong2 wasm_simd_load_extend(device const uchar* memory, ulong offset, uint source_bits, bool signed_lanes) {
    uint target_bits = source_bits * 2u;
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 128u / target_bits; ++i) {
        ulong part = wasm_simd_read(memory, offset + i * (source_bits / 8u), source_bits / 8u);
        if (signed_lanes) part = ulong(wasm_simd_signed(part, source_bits));
        result = wasm_simd_replace(result, target_bits, i, part);
    }
    return result;
}

inline ulong2 wasm_simd_splat(ulong value, uint bits) {
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 128u / bits; ++i) result = wasm_simd_replace(result, bits, i, value);
    return result;
}

// Operation numbers are selected by the Rust translator and are compile-time literals.
inline ulong2 wasm_simd_binary(ulong2 a, ulong2 b, uint bits, uint op) {
    ulong2 result = ulong2(0ul);
    ulong mask = wasm_simd_mask(bits);
    for (uint i = 0u; i < 128u / bits; ++i) {
        ulong x = wasm_simd_lane(a, bits, i);
        ulong y = wasm_simd_lane(b, bits, i);
        long sx = wasm_simd_signed(x, bits);
        long sy = wasm_simd_signed(y, bits);
        ulong v = 0ul;
        switch (op) {
            case 0u: v = x + y; break;
            case 1u: v = x - y; break;
            case 2u: v = x * y; break;
            case 3u: v = x == y ? mask : 0ul; break;
            case 4u: v = x != y ? mask : 0ul; break;
            case 5u: v = sx < sy ? mask : 0ul; break;
            case 6u: v = x < y ? mask : 0ul; break;
            case 7u: v = sx > sy ? mask : 0ul; break;
            case 8u: v = x > y ? mask : 0ul; break;
            case 9u: v = sx <= sy ? mask : 0ul; break;
            case 10u: v = x <= y ? mask : 0ul; break;
            case 11u: v = sx >= sy ? mask : 0ul; break;
            case 12u: v = x >= y ? mask : 0ul; break;
            case 13u: v = sx < sy ? x : y; break;
            case 14u: v = x < y ? x : y; break;
            case 15u: v = sx > sy ? x : y; break;
            case 16u: v = x > y ? x : y; break;
            case 17u: v = (x + y + 1ul) >> 1u; break;
            case 18u: {
                long sum = sx + sy;
                long low = -(1l << (bits - 1u));
                long high = (1l << (bits - 1u)) - 1l;
                v = ulong(clamp(sum, low, high));
                break;
            }
            case 19u: v = min(x + y, mask); break;
            case 20u: {
                long difference = sx - sy;
                long low = -(1l << (bits - 1u));
                long high = (1l << (bits - 1u)) - 1l;
                v = ulong(clamp(difference, low, high));
                break;
            }
            case 21u: v = x < y ? 0ul : x - y; break;
            default: break;
        }
        result = wasm_simd_replace(result, bits, i, v);
    }
    return result;
}

inline ulong2 wasm_simd_shift(ulong2 a, ulong amount, uint bits, uint op) {
    ulong2 result = ulong2(0ul);
    uint shift = uint(amount) & (bits - 1u);
    for (uint i = 0u; i < 128u / bits; ++i) {
        ulong x = wasm_simd_lane(a, bits, i);
        ulong v = op == 0u ? x << shift : op == 1u ? ulong(wasm_simd_signed(x, bits) >> shift) : x >> shift;
        result = wasm_simd_replace(result, bits, i, v);
    }
    return result;
}

inline ulong2 wasm_simd_unary(ulong2 a, uint bits, uint op) {
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 128u / bits; ++i) {
        ulong x = wasm_simd_lane(a, bits, i);
        long signed_x = wasm_simd_signed(x, bits);
        ulong v = op == 0u ? (signed_x < 0 ? 0ul - x : x) : 0ul - x;
        result = wasm_simd_replace(result, bits, i, v);
    }
    return result;
}

inline ulong wasm_simd_all_true(ulong2 value, uint bits) {
    for (uint i = 0u; i < 128u / bits; ++i)
        if (wasm_simd_lane(value, bits, i) == 0ul) return 0ul;
    return 1ul;
}

inline ulong wasm_simd_bitmask(ulong2 value, uint bits) {
    ulong result = 0ul;
    for (uint i = 0u; i < 128u / bits; ++i)
        result |= ((wasm_simd_lane(value, bits, i) >> (bits - 1u)) & 1ul) << i;
    return result;
}

inline ulong2 wasm_simd_popcnt(ulong2 value) {
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 16u; ++i)
        result = wasm_simd_replace(result, 8u, i, popcount(uint(wasm_simd_lane(value, 8u, i))));
    return result;
}

inline ulong2 wasm_simd_swizzle(ulong2 value, ulong2 indices) {
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 16u; ++i) {
        ulong lane = wasm_simd_lane(indices, 8u, i);
        result = wasm_simd_replace(result, 8u, i, lane < 16ul ? wasm_simd_lane(value, 8u, uint(lane)) : 0ul);
    }
    return result;
}

inline ulong wasm_simd_float_unary_lane(ulong x, uint bits, uint op) {
    if (bits == 32u) {
        uint value = uint(x);
        switch (op) {
            case 0u: return value & 0x7fffffffu;
            case 1u: return value ^ 0x80000000u;
            case 2u: return wasm_f32_sqrt(value);
            case 3u: return __softfloat64_cvt_f64_to_f32(wasm_f64_ceil(__softfloat64_cvt_f32_to_f64(value)), 0u);
            case 4u: return __softfloat64_cvt_f64_to_f32(wasm_f64_floor(__softfloat64_cvt_f32_to_f64(value)), 0u);
            case 5u: return __softfloat64_cvt_f64_to_f32(wasm_f64_trunc(__softfloat64_cvt_f32_to_f64(value)), 0u);
            case 6u: return __softfloat64_cvt_f64_to_f32(wasm_f64_nearest(__softfloat64_cvt_f32_to_f64(value)), 0u);
        }
    } else {
        switch (op) {
            case 0u: return x & 0x7ffffffffffffffful;
            case 1u: return x ^ 0x8000000000000000ul;
            case 2u: return __softfloat64_fsqrt(x, 0u);
            case 3u: return wasm_f64_ceil(x);
            case 4u: return wasm_f64_floor(x);
            case 5u: return wasm_f64_trunc(x);
            case 6u: return wasm_f64_nearest(x);
        }
    }
    return 0ul;
}

inline ulong2 wasm_simd_float_unary(ulong2 value, uint bits, uint op) {
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 128u / bits; ++i)
        result = wasm_simd_replace(result, bits, i, wasm_simd_float_unary_lane(wasm_simd_lane(value, bits, i), bits, op));
    return result;
}

inline ulong wasm_simd_float_binary_lane(ulong x, ulong y, uint bits, uint op) {
    ulong a = bits == 32u ? __softfloat64_cvt_f32_to_f64(uint(x)) : x;
    ulong b = bits == 32u ? __softfloat64_cvt_f32_to_f64(uint(y)) : y;
    ulong mask = wasm_simd_mask(bits);
    if (op == 12u) return __softfloat64_flt(b, a) ? y : x;
    if (op == 13u) return __softfloat64_flt(a, b) ? y : x;
    if (op >= 6u) {
        bool truth = false;
        switch (op) {
            case 6u: truth = __softfloat64_feq(a, b); break;
            case 7u: truth = !__softfloat64_feq(a, b); break;
            case 8u: truth = __softfloat64_flt(a, b); break;
            case 9u: truth = __softfloat64_fgt(a, b); break;
            case 10u: truth = __softfloat64_fle(a, b); break;
            case 11u: truth = __softfloat64_fge(a, b); break;
        }
        return truth ? mask : 0ul;
    }
    if (bits == 32u) {
        switch (op) {
            case 0u: return wasm_f32_add(uint(x), uint(y));
            case 1u: return wasm_f32_sub(uint(x), uint(y));
            case 2u: return wasm_f32_mul(uint(x), uint(y));
            case 3u: return wasm_f32_div(uint(x), uint(y));
            case 4u: return wasm_f32_min(uint(x), uint(y));
            case 5u: return wasm_f32_max(uint(x), uint(y));
        }
    } else {
        switch (op) {
            case 0u: return __softfloat64_fadd(x, y, 0u);
            case 1u: return __softfloat64_fsub(x, y, 0u);
            case 2u: return __softfloat64_fmul(x, y, 0u);
            case 3u: return __softfloat64_fdiv(x, y, 0u);
            case 4u: return wasm_f64_min(x, y);
            case 5u: return wasm_f64_max(x, y);
        }
    }
    return 0ul;
}

inline ulong2 wasm_simd_float_binary(ulong2 a, ulong2 b, uint bits, uint op) {
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 128u / bits; ++i)
        result = wasm_simd_replace(result, bits, i, wasm_simd_float_binary_lane(wasm_simd_lane(a, bits, i), wasm_simd_lane(b, bits, i), bits, op));
    return result;
}

inline ulong2 wasm_simd_extend(ulong2 value, uint source_bits, bool high, bool signed_lanes) {
    uint target_bits = source_bits * 2u;
    uint count = 128u / target_bits;
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < count; ++i) {
        ulong part = wasm_simd_lane(value, source_bits, i + (high ? count : 0u));
        if (signed_lanes) part = ulong(wasm_simd_signed(part, source_bits));
        result = wasm_simd_replace(result, target_bits, i, part);
    }
    return result;
}

inline ulong2 wasm_simd_narrow(ulong2 a, ulong2 b, uint source_bits, bool signed_lanes) {
    uint target_bits = source_bits / 2u;
    uint count = 128u / source_bits;
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < count * 2u; ++i) {
        ulong raw = wasm_simd_lane(i < count ? a : b, source_bits, i % count);
        ulong part;
        if (signed_lanes) {
            long low = -(1l << (target_bits - 1u));
            long high = (1l << (target_bits - 1u)) - 1l;
            part = ulong(clamp(wasm_simd_signed(raw, source_bits), low, high));
        } else {
            part = ulong(clamp(wasm_simd_signed(raw, source_bits), 0l, long(wasm_simd_mask(target_bits))));
        }
        result = wasm_simd_replace(result, target_bits, i, part);
    }
    return result;
}

inline ulong2 wasm_simd_extmul(ulong2 a, ulong2 b, uint source_bits, bool high, bool signed_lanes) {
    uint target_bits = source_bits * 2u;
    uint count = 128u / target_bits;
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < count; ++i) {
        uint lane = i + (high ? count : 0u);
        ulong x = wasm_simd_lane(a, source_bits, lane);
        ulong y = wasm_simd_lane(b, source_bits, lane);
        ulong part = signed_lanes ? ulong(wasm_simd_signed(x, source_bits) * wasm_simd_signed(y, source_bits)) : x * y;
        result = wasm_simd_replace(result, target_bits, i, part);
    }
    return result;
}

inline ulong2 wasm_simd_pairwise(ulong2 value, uint source_bits, bool signed_lanes) {
    uint target_bits = source_bits * 2u;
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 128u / target_bits; ++i) {
        ulong x = wasm_simd_lane(value, source_bits, i * 2u);
        ulong y = wasm_simd_lane(value, source_bits, i * 2u + 1u);
        ulong part = signed_lanes ? ulong(wasm_simd_signed(x, source_bits) + wasm_simd_signed(y, source_bits)) : x + y;
        result = wasm_simd_replace(result, target_bits, i, part);
    }
    return result;
}

inline ulong2 wasm_simd_dot(ulong2 a, ulong2 b) {
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 4u; ++i) {
        long x0 = wasm_simd_signed(wasm_simd_lane(a, 16u, i * 2u), 16u);
        long x1 = wasm_simd_signed(wasm_simd_lane(a, 16u, i * 2u + 1u), 16u);
        long y0 = wasm_simd_signed(wasm_simd_lane(b, 16u, i * 2u), 16u);
        long y1 = wasm_simd_signed(wasm_simd_lane(b, 16u, i * 2u + 1u), 16u);
        result = wasm_simd_replace(result, 32u, i, ulong(x0 * y0 + x1 * y1));
    }
    return result;
}

inline ulong2 wasm_simd_q15mulr(ulong2 a, ulong2 b) {
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 8u; ++i) {
        long x = wasm_simd_signed(wasm_simd_lane(a, 16u, i), 16u);
        long y = wasm_simd_signed(wasm_simd_lane(b, 16u, i), 16u);
        long part = min((x * y + 16384l) >> 15u, 32767l);
        result = wasm_simd_replace(result, 16u, i, ulong(part));
    }
    return result;
}

inline ulong2 wasm_simd_convert(ulong2 value, uint op) {
    ulong2 result = ulong2(0ul);
    uint count = (op == 4u || op == 5u || op == 6u || op == 7u || op == 8u || op == 9u) ? 2u : 4u;
    for (uint i = 0u; i < count; ++i) {
        ulong x = wasm_simd_lane(value, (op == 4u || op == 5u || op == 8u) ? 64u : 32u, i);
        ulong part = 0ul;
        uint out_bits = (op == 6u || op == 7u || op == 9u) ? 64u : 32u;
        uint ignored_trap = 0u;
        switch (op) {
            case 0u: part = wasm_f64_to_int(__softfloat64_cvt_f32_to_f64(uint(x)), 32u, true, true, ignored_trap); break;
            case 1u: part = wasm_f64_to_int(__softfloat64_cvt_f32_to_f64(uint(x)), 32u, false, true, ignored_trap); break;
            case 2u: part = as_type<uint>(float(as_type<int>(uint(x)))); break;
            case 3u: part = as_type<uint>(float(uint(x))); break;
            case 4u: part = wasm_f64_to_int(x, 32u, true, true, ignored_trap); break;
            case 5u: part = wasm_f64_to_int(x, 32u, false, true, ignored_trap); break;
            case 6u: part = __softfloat64_cvt_i64_to_f64(long(as_type<int>(uint(x))), 0u); break;
            case 7u: part = __softfloat64_cvt_u64_to_f64(ulong(uint(x)), 0u); break;
            case 8u: part = __softfloat64_cvt_f64_to_f32(x, 0u); break;
            case 9u: part = __softfloat64_cvt_f32_to_f64(uint(x)); break;
        }
        result = wasm_simd_replace(result, out_bits, i, part);
    }
    return result;
}

inline ulong2 wasm_simd_relaxed_madd(ulong2 a, ulong2 b, ulong2 c, uint bits, bool negate_a) {
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 128u / bits; ++i) {
        ulong x = wasm_simd_lane(a, bits, i);
        if (negate_a) x ^= 1ul << (bits - 1u);
        ulong y = wasm_simd_lane(b, bits, i);
        ulong z = wasm_simd_lane(c, bits, i);
        ulong product = wasm_simd_float_binary_lane(x, y, bits, 2u);
        result = wasm_simd_replace(result, bits, i, wasm_simd_float_binary_lane(product, z, bits, 0u));
    }
    return result;
}

inline ulong2 wasm_simd_relaxed_dot16(ulong2 a, ulong2 b) {
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 8u; ++i) {
        long sum = 0l;
        for (uint j = 0u; j < 2u; ++j) {
            uint lane = i * 2u + j;
            sum += wasm_simd_signed(wasm_simd_lane(a, 8u, lane), 8u)
                 * wasm_simd_signed(wasm_simd_lane(b, 8u, lane), 8u);
        }
        result = wasm_simd_replace(result, 16u, i, ulong(clamp(sum, -32768l, 32767l)));
    }
    return result;
}

inline ulong2 wasm_simd_relaxed_dot_add(ulong2 a, ulong2 b, ulong2 c) {
    ulong2 pairs = wasm_simd_relaxed_dot16(a, b);
    ulong2 result = ulong2(0ul);
    for (uint i = 0u; i < 4u; ++i) {
        long x = wasm_simd_signed(wasm_simd_lane(pairs, 16u, i * 2u), 16u);
        long y = wasm_simd_signed(wasm_simd_lane(pairs, 16u, i * 2u + 1u), 16u);
        ulong z = wasm_simd_lane(c, 32u, i);
        result = wasm_simd_replace(result, 32u, i, ulong(x + y) + z);
    }
    return result;
}

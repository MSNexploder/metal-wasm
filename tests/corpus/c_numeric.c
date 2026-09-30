#include <stdint.h>

static uint32_t rotate(uint32_t value, unsigned shift) {
    return (value << shift) | (value >> (32 - shift));
}

__attribute__((visibility("default")))
uint32_t calculate(uint32_t input) {
    uint64_t wide = ((uint64_t)input << 32) | (input ^ 0x85ebca6bu);
    int64_t signed_value = (int64_t)(int32_t)input;
    double fraction = (double)(int32_t)input / 17.0;
    uint32_t words[32];
    for (unsigned i = 0; i < 32; ++i) {
        wide = wide * UINT64_C(0xd6e8feb86659fd93) + i;
        words[i] = (uint32_t)(wide >> 32) ^ input;
    }
    uint32_t result = input;
    for (unsigned i = 0; i < 48; ++i) {
        switch ((result ^ i) & 3u) {
        case 0: result = rotate(result, 5) + words[i & 31]; break;
        case 1: result ^= (uint32_t)(wide / (i + 1)); break;
        case 2: result += (uint32_t)(signed_value / (int64_t)(i + 1)); break;
        default: result = result * 7 + (uint32_t)(int32_t)fraction; break;
        }
        fraction = fraction * 0.5 + (double)i;
    }
    return result ^ (uint32_t)wide;
}

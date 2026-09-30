#include <stdint.h>

struct Cell { uint32_t value; uint32_t salt; };
static Cell cells[32];
static uint32_t calls;
static uint32_t add(uint32_t value, uint32_t salt) { return value + salt; }
static uint32_t mix(uint32_t value, uint32_t salt) {
    return ((value << 7) | (value >> 25)) ^ salt;
}
static uint32_t multiply(uint32_t value, uint32_t salt) { return value * (salt | 1); }
using Operation = uint32_t (*)(uint32_t, uint32_t);
static Operation operations[] = {add, mix, multiply};

extern "C" __attribute__((visibility("default")))
uint32_t calculate(uint32_t input) {
    ++calls;
    for (unsigned i = 0; i < 32; ++i) {
        cells[i].value = input + i * 0x9e3779b9u;
        cells[i].salt = calls ^ (i * 0x85ebca6bu);
    }
    uint32_t value = input ^ calls;
    for (unsigned i = 0; i < 64; ++i) {
        Cell &cell = cells[(value ^ i) & 31];
        // Volatile lookup keeps a real Wasm indirect call in optimized builds.
        Operation operation = *reinterpret_cast<Operation volatile *>(&operations[(value >> 8) % 3]);
        value = operation(value ^ cell.value, cell.salt);
        cell.value = value;
    }
    return value + calls;
}

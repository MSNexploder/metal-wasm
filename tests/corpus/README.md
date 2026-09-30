# Compiled Wasm corpus

These `.wasm` files are built from adjacent Rust, C, and C++ sources.
The checked-in builds use Rust 1.98.0 and Apple Clang 21.0.0.
`scalar` and `memory` have no imports. The test instantiates them separately
for each input in Wasmtime to match the GPU's independent per-lane state.
`host_import` deliberately imports `env.host_mix`; it demonstrates an
unsupported host dependency when no Wasm provider is registered.
`persistent` keeps a mutable Rust static across calls. Its test compares each
Metal lane with its own persistent Wasmtime instance through two exports.

Rebuild with the `wasm32-wasip1` target installed:

```sh
for name in scalar memory persistent; do
  rustc --target wasm32-wasip1 --crate-type cdylib \
    -C opt-level=3 -C panic=abort -C link-arg=-zstack-size=65536 \
    "tests/corpus/$name.rs" -o "tests/corpus/$name.wasm"
done
rustc --target wasm32-wasip1 --crate-type cdylib \
  -C opt-level=3 -C panic=abort -C link-arg=-zstack-size=65536 \
  -C link-arg=--allow-undefined \
  tests/corpus/host_import.rs -o tests/corpus/host_import.wasm
```

The checked-in binaries keep the compatibility tests runnable without a Wasm
compiler installed. Add compiled programs here only when the source and build
command can be kept with the binary.

## Additional compiler workloads

`c_numeric_o0` and `c_numeric_o3` cover arrays, switches, signed and unsigned
64-bit division, and double arithmetic. `cpp_dispatch_o0` and `cpp_dispatch_o3`
cover struct arrays, a mutable static, and a volatile function-pointer table.
Each C++ lane is also compared with a persistent Wasmtime instance across calls.
The freestanding C++ workload does not require a C++ runtime.
`rust_numeric` covers `u128` arithmetic, rotations, conversions, and `f64`.
`rust_simd` uses wasm32 intrinsics for vector multiplication, addition, shuffles,
bit selection, and shifts. Its compiled module retains SIMD instructions.

Build the Rust workloads with the installed `wasm32-wasip1` target:

```sh
for name in rust_numeric rust_simd; do
  rustc --target wasm32-wasip1 --crate-type cdylib \
    -C opt-level=3 -C panic=abort -C target-feature=+simd128 \
    -C link-arg=-zstack-size=65536 \
    "tests/corpus/$name.rs" -o "tests/corpus/$name.wasm"
done
```

Build C and C++ on an Apple Silicon Mac with Clang's wasm32 backend and the
Rust toolchain's `rust-lld`. `DYLD_LIBRARY_PATH` supplies its adjacent LLVM library.
Compiler and linker versions affect the binary bytes.

```sh
wasm_toolchain="$(rustc --print sysroot)"
wasm_linker="$wasm_toolchain/lib/rustlib/aarch64-apple-darwin/bin/rust-lld"
for level in 0 3; do
  clang --target=wasm32-unknown-unknown -ffreestanding -fno-builtin \
    -O"$level" -c tests/corpus/c_numeric.c \
    -o "/tmp/metal-wasm-c-numeric-o$level.o"
  DYLD_LIBRARY_PATH="$wasm_toolchain/lib" "$wasm_linker" -flavor wasm \
    --no-entry --export=calculate -z stack-size=65536 \
    "/tmp/metal-wasm-c-numeric-o$level.o" \
    -o "tests/corpus/c_numeric_o$level.wasm"
  clang --target=wasm32-unknown-unknown -ffreestanding -fno-builtin \
    -fno-exceptions -fno-rtti -O"$level" \
    -c tests/corpus/cpp_dispatch.cpp \
    -o "/tmp/metal-wasm-cpp-dispatch-o$level.o"
  DYLD_LIBRARY_PATH="$wasm_toolchain/lib" "$wasm_linker" -flavor wasm \
    --no-entry --export=calculate -z stack-size=65536 \
    "/tmp/metal-wasm-cpp-dispatch-o$level.o" \
    -o "tests/corpus/cpp_dispatch_o$level.wasm"
done
```

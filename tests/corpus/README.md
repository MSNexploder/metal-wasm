# Compiled Wasm corpus

These `.wasm` files are built from the adjacent Rust sources with Rust 1.98.0.
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

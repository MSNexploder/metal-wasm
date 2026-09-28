# metal-wasm

Run many independent invocations of one WebAssembly function on a Metal GPU. Each input value gets its own Metal thread and produces one output value.

This first version accepts a `.wasm` module with an exported `(param i32) (result i32)` function. It supports straight-line `i32` code: constants, locals, drop, equality, add, subtract, multiply, bitwise operations, shifts, and rotations. Other instructions and module sections return an `Unsupported` error. There is no linear memory, imports, control flow, or shared state yet. Values use `u32` in Rust to preserve the raw WebAssembly `i32` bits.

```rust
use metal_wasm::Kernel;

let wasm = std::fs::read("function.wasm")?;
let kernel = Kernel::new(&wasm, "calculate")?;
let output = kernel.run(&[1, 2, 3])?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Requires macOS with a Metal device and the Rust toolchain. Run `cargo test` to validate GPU execution. The kernel compiles once and can run repeatedly with different input batches.

## Performance comparison

Run `cargo run --release --example compare` to compare Metal, Wasmtime, Wasmi, and optimized native Rust at 1, 1,000, 1,000,000, and 10,000,000 invocations. The example uses the same `x * 7 + 3` function for every path, checks every result, and prints median elapsed times in microseconds. It takes 31 samples of 100 runs for 1 invocation, 11 samples of 10 runs for 1,000, and 3 single-run samples for each larger count. Small runs are repeated inside each timed sample to reduce clock overhead.

Run `cargo run --release --example compare -- --metal-native-100m` to compare Metal and native Rust at 100,000,000 invocations. This mode takes 3 single-run samples and skips the Wasm engines.

Run `cargo run --release --example compare -- --mix` for a compute-heavy comparison at 1,000, 1,000,000, 10,000,000, and 100,000,000 invocations. Each input goes through 32 rounds of rotate, XOR, multiply, and add in Wasm on Metal, Wasmtime, Wasmi, and native Rust. The two CPU Wasm engines stop at 10,000,000 inputs. This mode reports Metal end-to-end time and Metal command-buffer GPU time separately. The GPU time comes from Metal's `GPUStartTime` and `GPUEndTime` after command completion; it excludes CPU buffer copies and submission. It includes GPU memory access and kernel execution. The benchmark fails if GPU timestamps are unavailable.

The timed region includes result allocation and execution. Metal also includes input transfer, command submission, and result transfer. Module compilation, instantiation, warmup, input generation, and result verification are outside the timed region. Wasmtime and Wasmi reuse one stateless instance and make one typed host call per input; Metal uses a single batched dispatch. These figures compare these APIs for this workload, not peak engine or GPU throughput.

# metal-wasm

Run many independent invocations of one WebAssembly function on a Metal GPU. Each input value gets its own Metal thread and produces one output value.

This version accepts a `.wasm` module with an exported `(param i32) (result i32)` function. It supports all self-contained scalar `i32` instructions: constants, locals, drop, select, comparisons, arithmetic, division, remainder, bit counts, bitwise operations, shifts, rotations, and sign extension. It supports `block`, `if`/`else`, `loop`, `br`, `br_if`, `br_table`, `return`, `nop`, and `unreachable`. Blocks can use type-index signatures with multiple `i32` parameters and results. If an invocation traps, `run` returns `Error::Trap` with the first input index and the reason; results for that batch are discarded. Other instructions and module sections return an `Unsupported` error. There is no linear memory, imports, calls, or shared state yet. Values use `u32` in Rust to preserve the raw WebAssembly `i32` bits.

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

Run `cargo run --release --example compare -- --control` for the same counts with a 32-iteration Wasm loop and an `if`/`else` in every iteration. The `grouped` pattern places even inputs before odd inputs in groups of 32, so neighboring lanes usually take the same branch. The `mixed` pattern interleaves those same even and odd values, so neighboring lanes disagree. Both patterns execute the same arithmetic and number of iterations overall; only input order changes. Each path runs on Metal, Wasmtime, Wasmi, and native Rust, with the CPU Wasm engines omitted at 100,000,000 inputs. This mode uses three samples and checks every result. The `pattern` column identifies the input pattern; `metal_gpu_us` includes GPU buffer access and kernel execution.

The timed region includes result allocation and execution. Metal also includes input transfer, command submission, and result transfer. Module compilation, instantiation, warmup, input generation, and result verification are outside the timed region. Wasmtime and Wasmi reuse one stateless instance and make one typed host call per input; Metal uses a single batched dispatch. These figures compare these APIs for this workload, not peak engine or GPU throughput.

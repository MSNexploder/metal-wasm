# metal-wasm

Run many independent invocations of one WebAssembly function on a Metal GPU. Each invocation gets its own Metal thread and its own module state.

This version accepts exported functions with any number of `i32`, `i64`, `f32`, and `f64` parameters and results. `Kernel::run` retains the fast `(i32) -> i32` interface; `Kernel::run_values` accepts a row of raw `Value` bit patterns per invocation. It supports scalar numeric operations, conversions, structured control flow, direct and indirect calls, and recursion up to 32 call frames by default. Deeper calls trap. Type-index blocks and calls can pass multiple values, including internal `funcref` values.

Each module can own or import one unshared 32-bit linear memory, active and passive data segments, globals, and a start function. Supported memory instructions include loads, stores, `memory.size`, `memory.grow`, `memory.copy`, `memory.fill`, `memory.init`, and `data.drop`. The default initial memory limit is 16 MiB per module instance. Growth can add at most four pages by default, subject to the declared maximum and the per-lane memory limit; excess growth returns `-1`. `KernelLimits` lets callers raise the memory limit, growth reserve, table growth reserve, and call depth when compiling. Larger reserves use more memory per lane and can reduce batch size. Large batches use consecutive dispatches sized for a 64 MiB buffer budget; per-lane state buffers are reused and reset between dispatches. A single invocation or shared buffer that exceeds a Metal buffer limit returns `Unsupported`. If an invocation traps, the whole run returns `Error::Trap` with its index in the original input and reason; results are discarded.

Each module can own or import one unshared 32-bit `funcref` table with at most 65,536 initial entries. Active, passive, and declared element segments are supported. `table.get`, `table.set`, `table.size`, `table.grow`, `table.fill`, `table.copy`, `table.init`, `elem.drop`, and `ref.null`, `ref.func`, and `ref.is_null` operate on per-lane table state. Growth can add at most 256 entries by default, subject to the declared maximum and the 65,536-entry limit. Unchanged tables share an initialized buffer between lanes. `call_indirect` checks the slot and function signature at runtime.

`Kernel::new_with_imports(wasm, export, providers)` links function, memory, table, and global imports to exports from supplied Wasm provider modules. `providers` is a slice of `(namespace, wasm_bytes)` pairs. Each namespace has its own module state per lane; imported resources refer to the provider's state. Active segments and start functions run in dependency order. Imported functions run on the GPU. `Kernel::new_with_imports_and_limits` combines linking with custom limits. Linked module state must fit Metal's 31 buffer slots. Arbitrary host callbacks remain unsupported.

`Kernel::run_values_with_memory(inputs, memory_export, range)` returns function results and the requested byte range from an exported memory for each lane. The range must fit every lane's memory after execution. It can read a memory owned by the module or imported from a provider. Copying large ranges adds host transfer time.

Metal has no native `f64`, and this Mac flushes native `f32` subnormals. Both types use GPU software arithmetic on raw bits. The `f64` primitives come from the vendored [metal-softfloat](https://github.com/guyfischman/metal-softfloat) source under the licenses in `licenses/`. Floating-point operations cost substantially more GPU work than native `i32` operations.

Public reference-typed parameters and results, shared memory, SIMD, and host interfaces such as WASI remain unsupported.

```rust
use metal_wasm::Kernel;

let wasm = std::fs::read("function.wasm")?;
let kernel = Kernel::new(&wasm, "calculate")?;
let output = kernel.run(&[1, 2, 3])?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Requires macOS with a Metal device and the Rust toolchain. Run `cargo test` to validate GPU execution. The kernel compiles once and can run repeatedly with different input batches. `run_timed` sums GPU execution time across dispatches; it excludes host-side buffer copies and submission.

## Compatibility checks

Run `cargo run --example compat -- MODULE.wasm EXPORT I32_INPUT ...` with chosen inputs to check an exported `(i32) -> i32` function against fresh Wasmtime instances. It reports an unsupported module or signature with the reason, accepts matching traps, and fails if Metal and Wasmtime disagree. For example, `cargo run --example compat -- tests/corpus/memory.wasm calculate 0 1 42` checks a compiled Rust memory workload. Run only terminating functions; the GPU runner has no execution timeout.

`cargo test --test compiled_corpus` checks two supported compiled Rust modules against Wasmtime and checks that a third compiled module reports its unsupported host import. Their sources, binaries, and build commands are in `tests/corpus/`. `cargo test --test spec_runtime` runs 5,558 return and trap assertions from pinned official `i32`, `i64`, `f32_cmp`, and `f64_cmp` spec scripts. Invalid and malformed module assertions are counted but excluded from this execution harness. Runtime tests also compare memory, table, import, and initialization behavior with Wasmtime. These checks do not establish full Wasm conformance.

## Performance comparison

Run `cargo run --release --example compare` to compare Metal, Wasmtime, Wasmi, and optimized native Rust at 1, 1,000, 1,000,000, and 10,000,000 invocations. The example uses the same `x * 7 + 3` function for every path, checks every result, and prints median elapsed times in microseconds. It takes 31 samples of 100 runs for 1 invocation, 11 samples of 10 runs for 1,000, and 3 single-run samples for each larger count. Small runs are repeated inside each timed sample to reduce clock overhead.

Run `cargo run --release --example compare -- --metal-native-100m` to compare Metal and native Rust at 100,000,000 invocations. This mode takes 3 single-run samples and skips the Wasm engines.

Run `cargo run --release --example compare -- --mix` for a compute-heavy comparison at 1,000, 1,000,000, 10,000,000, and 100,000,000 invocations. Each input goes through 32 rounds of rotate, XOR, multiply, and add in Wasm on Metal, Wasmtime, Wasmi, and native Rust. The two CPU Wasm engines stop at 10,000,000 inputs. This mode reports Metal end-to-end time and Metal command-buffer GPU time separately. The GPU time comes from Metal's `GPUStartTime` and `GPUEndTime` after command completion; it excludes CPU buffer copies and submission. It includes GPU memory access and kernel execution. The benchmark fails if GPU timestamps are unavailable.

Run `cargo run --release --example compare -- --control` for the same counts with a 32-iteration Wasm loop and an `if`/`else` in every iteration. The `grouped` pattern places even inputs before odd inputs in groups of 32, so neighboring lanes usually take the same branch. The `mixed` pattern interleaves those same even and odd values, so neighboring lanes disagree. Both patterns execute the same arithmetic and number of iterations overall; only input order changes. Each path runs on Metal, Wasmtime, Wasmi, and native Rust, with the CPU Wasm engines omitted at 100,000,000 inputs. This mode uses three samples and checks every result. The `pattern` column identifies the input pattern; `metal_gpu_us` includes GPU buffer access and kernel execution.

Run `cargo run --release --example compare -- --suite` for seven workloads. The original `dispatch_i64`, `float_f64`, and `stateful_memory` cases remain. `bulk_start` uses a start function and bulk memory operations; `recursive_calls` uses variable-depth recursion; `mutable_table` uses passive elements, table updates, and indirect calls; `imported_calls` links a separate Wasm provider module. Their definitions are in `examples/workloads/`. Each workload is checked against optimized native Rust on all four paths. This mode reports the median of three single-run samples at 1, 1,000, 100,000, and 1,000,000 invocations, except the two memory workloads, which use 1, 1,000, 4,096, and 10,000 to exercise bounded dispatches.

The timed region includes result allocation and execution. Metal also includes per-lane state initialization, input transfer, command submission, and result transfer. Module compilation, CPU engine instantiation, warmup, input generation, and result verification are outside the timed region. Wasmtime and Wasmi reuse one instance per module and make one typed host call per input; the stateful workload resets its touched memory and global before returning. Metal batches one or more dispatches and runs each start function per lane. Thus `bulk_start` includes per-lane start work on Metal, while CPU start work occurs outside the timed region. These figures compare these APIs for these workloads, not peak engine or GPU throughput.

# metal-wasm

Run many independent invocations of one WebAssembly function on a Metal GPU. Each invocation gets its own Metal thread and its own module state.

This version accepts exported functions with any number of `i32`, `i64`, `f32`, `f64`, `v128`, and function reference parameters and results. `Kernel::run` retains the fast `(i32) -> i32` interface; `Kernel::run_values` accepts a row of raw `Value` bit patterns per invocation. `Value::V128` stores 16 bytes in Wasm lane order. It supports scalar numeric operations, SIMD operations, conversions, structured control flow, direct and indirect calls, and `return_call`, `return_call_indirect`, and `return_call_ref`. Tail calls reuse a shader call frame. Ordinary recursion is limited to 32 call frames by default; deeper calls trap. Larger call-depth limits can exceed the device’s stack capacity. Type-index blocks and calls can pass multiple values, including nullable and non-nullable typed function references. Function-only recursive type groups and declared function subtyping are supported.

Each module can own or import multiple unshared linear memories with 32-bit or 64-bit addresses, active and passive data segments, globals, and a start function. Supported memory instructions include loads, stores, `memory.size`, `memory.grow`, `memory.copy`, `memory.fill`, `memory.init`, and `data.drop`. Initializers support `i32` and `i64` add, subtract, multiply, and reads of earlier immutable globals. The default total memory limit is 16 MiB per module instance. Growth can add at most four pages per memory by default, subject to declared maxima and that total limit; excess growth returns `-1`. `KernelLimits` lets callers raise the memory limit, growth reserve, table growth reserve, call depth, and exception record limit when compiling. Each growable memory reserves GPU buffer space for its possible growth, even when the total limit prevents every memory from growing in one lane. Larger reserves can reduce batch size. Large batches use consecutive dispatches sized for a 64 MiB buffer budget; per-lane state buffers are reused and reset between dispatches. A single invocation or shared buffer that exceeds a Metal buffer limit returns `Unsupported`. If an invocation traps, the whole run returns `Error::Trap` with its index in the original input and reason; results are discarded.

Each module can own or import multiple unshared function reference tables, including typed and non-nullable element types, with 32-bit or 64-bit indices. Each table has at most 65,536 initial entries. Active, passive, and declared element segments are supported. `table.get`, `table.set`, `table.size`, `table.grow`, `table.fill`, `table.copy`, `table.init`, `elem.drop`, and `ref.null`, `ref.func`, `ref.is_null`, `ref.as_non_null`, `br_on_null`, and `br_on_non_null` operate on per-lane table state. Growth can add at most 256 entries by default, subject to the declared maximum and the 65,536-entry limit. Unchanged tables share an initialized buffer between lanes. `call_indirect` checks the slot and nominal function type at runtime. `call_ref` dispatches a typed function reference.

`Kernel::new_with_imports(wasm, export, providers)` links function, memory, table, global, and exception tag imports to exports from supplied Wasm provider modules. `providers` is a slice of `(namespace, wasm_bytes)` pairs. Each namespace has its own module state per lane; imported resources refer to the provider's state. Active segments and start functions run in dependency order. Imported functions run on the GPU. `Kernel::new_with_imports_and_limits` combines linking with custom limits. Linked module state must fit Metal's 31 buffer slots. Arbitrary host callbacks remain unsupported.

`Kernel::run_values_with_memory(inputs, memory_export, range)` returns function results and the requested byte range from an exported memory for each lane. The range must fit every lane's memory after execution. It can read a memory owned by the module or imported from a provider. Copying large ranges adds host transfer time.

`Kernel::run_values_with_memory_io(inputs, memory_export, input_offset, memory_inputs, output_range)` also copies a distinct byte slice into each lane's exported memory after data segments and start functions run, immediately before the exported call. Supply one equally sized byte slice per invocation. The input range must fit the memory's initial size. Use `0..0` as the output range when no memory readback is needed.

Metal has no native `f64`, and this Mac flushes native `f32` subnormals. Both types use GPU software arithmetic on raw bits. The `f64` primitives come from the vendored [metal-softfloat](https://github.com/guyfischman/metal-softfloat) source under the licenses in `licenses/`. Floating-point operations cost substantially more GPU work than native `i32` operations.

`Kernel::function_reference(export)` returns an opaque handle for a root-module function export. Pass it as `Value::FuncRef(Some(handle))`; use `Value::FuncRef(None)` for null. `run_values` also returns function handles. Handles belong to one compiled kernel and cannot cross kernel boundaries. Input checks enforce reference nullability and declared function subtyping.

Exception handling supports tags, `try_table`, `throw`, `throw_ref`, `catch`, `catch_ref`, `catch_all`, and `catch_all_ref`. Calls propagate exceptions to the nearest matching handler. Tail calls remove the caller’s handlers. Traps cannot be caught. Exception references can be kept in locals or globals and rethrown. Each lane retains at most 64 immutable exception records per invocation by default; `KernelLimits::exception_records` changes this limit. Handled records remain allocated so retained references stay valid. Exhaustion traps with `exception record limit exceeded`. An escaping exception returns `Error::UncaughtException` with the original input index. Provider start exceptions stop initialization.

GC objects, public external or exception reference values, shared memory, and host interfaces such as WASI remain unsupported. Function type groups that contain GC object or continuation types are rejected.

```rust
use metal_wasm::Kernel;

let wasm = std::fs::read("function.wasm")?;
let kernel = Kernel::new(&wasm, "calculate")?;
let output = kernel.run(&[1, 2, 3])?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Requires macOS with a Metal device and the Rust toolchain. Run `cargo test` to validate GPU execution. The kernel compiles once and can run repeatedly with different input batches. `run_timed` sums GPU execution time across dispatches; it excludes host-side buffer copies and submission.

## Compatibility checks

Run `cargo run --example compat -- MODULE.wasm EXPORT I32_INPUT ...` with chosen inputs to check an exported `(i32) -> i32` function against fresh Wasmtime instances. It reports an unsupported module or signature with the reason, accepts matching traps and uncaught exceptions, and fails if Metal and Wasmtime disagree. For example, `cargo run --example compat -- tests/corpus/memory.wasm calculate 0 1 42` checks a compiled Rust memory workload. Run only terminating functions; the GPU runner has no execution timeout.

`cargo test --test compiled_corpus` checks two supported compiled Rust modules against Wasmtime and checks that a third compiled module reports its unsupported host import. Their sources, binaries, and build commands are in `tests/corpus/`. `cargo test --test spec_runtime` checks pinned official scalar, SIMD, typed function reference, exception, tail-call, extended-constant, multiple-memory, and memory64 assertions. The harness replays successful actions within one lane for `ref_func.wast` and the supported select assertions. Other scripts that require persistent instance state remain outside the harness; see `tests/spec/README.md`. Runtime tests also compare memory, table, import, and initialization behavior with Wasmtime. These checks do not establish full Wasm conformance.

## Performance comparison

Run `cargo run --release --example compare` to compare Metal, Wasmtime, Wasmi, and optimized native Rust at 1, 1,000, 1,000,000, and 10,000,000 invocations. The example uses the same `x * 7 + 3` function for every path, checks every result, and prints median elapsed times in microseconds. It takes 31 samples of 100 runs for 1 invocation, 11 samples of 10 runs for 1,000, and 3 single-run samples for each larger count. Small runs are repeated inside each timed sample to reduce clock overhead.

Run `cargo run --release --example compare -- --metal-native-100m` to compare Metal and native Rust at 100,000,000 invocations. This mode takes 3 single-run samples and skips the Wasm engines.

Run `cargo run --release --example compare -- --mix` for a compute-heavy comparison at 1,000, 1,000,000, 10,000,000, and 100,000,000 invocations. Each input goes through 32 rounds of rotate, XOR, multiply, and add in Wasm on Metal, Wasmtime, Wasmi, and native Rust. The two CPU Wasm engines stop at 10,000,000 inputs. This mode reports Metal end-to-end time and Metal command-buffer GPU time separately. The GPU time comes from Metal's `GPUStartTime` and `GPUEndTime` after command completion; it excludes CPU buffer copies and submission. It includes GPU memory access and kernel execution. The benchmark fails if GPU timestamps are unavailable.

Run `cargo run --release --example compare -- --control` for the same counts with a 32-iteration Wasm loop and an `if`/`else` in every iteration. The `grouped` pattern places even inputs before odd inputs in groups of 32, so neighboring lanes usually take the same branch. The `mixed` pattern interleaves those same even and odd values, so neighboring lanes disagree. Both patterns execute the same arithmetic and number of iterations overall; only input order changes. Each path runs on Metal, Wasmtime, Wasmi, and native Rust, with the CPU Wasm engines omitted at 100,000,000 inputs. This mode uses three samples and checks every result. The `pattern` column identifies the input pattern; `metal_gpu_us` includes GPU buffer access and kernel execution.

Run `cargo run --release --example compare -- --suite` for eleven workloads. The original `dispatch_i64`, `float_f64`, and `stateful_memory` cases remain. `bulk_start` uses a start function and bulk memory operations; `recursive_calls` uses variable-depth recursion; `mutable_table` uses passive elements, table updates, and indirect calls; `imported_calls` links a separate Wasm provider module; `linked_resources` imports and mutates provider memory, a global, and a table. `simd_i32x4` runs 32 rounds of four-lane integer arithmetic and a lane reduction. `typed_exceptions` runs 32 rounds of typed function dispatch, divergent branches, and payload throws and catches. `tail_multi_resource` combines tail calls, extended constants, two memories, a 64-bit memory, and a 64-bit table. Run `cargo run --release --example compare -- --simd`, `--tail-multi-resource`, or `--typed-exceptions` to measure one workload. Their definitions are in `examples/workloads/`. Each workload is checked against optimized native Rust and Wasmtime; Wasmi is omitted for the SIMD and new feature workloads because this version does not support their Wasm features. This mode reports the median of three single-run samples at 1, 1,000, 100,000, and 1,000,000 invocations, except the four memory workloads, which use 1, 1,000, 4,096, and 10,000 to exercise bounded dispatches.

The timed region includes result allocation and execution. Metal also includes per-lane state initialization, input transfer, command submission, and result transfer. Module compilation, CPU engine instantiation, warmup, input generation, and result verification are outside the timed region. Wasmtime and Wasmi reuse one instance per module and make one typed host call per input; the stateful and linked-resource workloads reset their touched state for each call. Metal batches one or more dispatches and runs each start function per lane. Thus `bulk_start` includes per-lane start work on Metal, while CPU start work occurs outside the timed region. These figures compare these APIs for these workloads, not peak engine or GPU throughput.

## Recorded benchmark results

The [full suite results](examples/benchmark-results.csv) were measured on 2026-09-29 with an Apple M1 Max, 64 GB RAM, macOS 27.0, and a release build. Each row reports the median of three samples. The new workload produced these times in milliseconds:

| Invocations | Metal total | Metal GPU | Wasmtime | Native Rust |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.33546 | 0.06788 | 0.00221 | 0.00008 |
| 1,000 | 0.36154 | 0.11171 | 2.37525 | 0.06308 |
| 100,000 | 1.67938 | 1.01913 | 244.14717 | 6.22700 |
| 1,000,000 | 10.11383 | 7.76488 | 2435.34258 | 62.58004 |

Wasmtime 49.0.1 uses its `gc-null` collector in this build; the enabled `gc` feature permits typed references and exceptions. Wasmi is omitted for this workload because it does not support these features in this build. Native Rust uses equivalent arithmetic and recovery branches. At one million inputs, Metal total time is about 6.2× faster than native Rust. At 1 and 1,000 inputs, native Rust is faster. Metal GPU time includes kernel work and GPU memory access. The API and timing differences described above apply.

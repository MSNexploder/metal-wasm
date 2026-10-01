# metal-wasm

Run many independent invocations of one WebAssembly function on a Metal GPU. Each invocation gets its own Metal thread and its own module state.

This version accepts exported functions with any number of `i32`, `i64`, `f32`, `f64`, `v128`, and function, external, exception, or GC reference parameters and results. `Kernel::run` retains the fast `(i32) -> i32` interface; `Kernel::run_values` accepts a row of raw `Value` bit patterns per invocation. `Value::V128` stores 16 bytes in Wasm lane order. It supports scalar numeric operations, SIMD operations, conversions, structured control flow, direct and indirect calls, and `return_call`, `return_call_indirect`, and `return_call_ref`. Wide arithmetic supports `i64.add128`, `i64.sub128`, `i64.mul_wide_s`, and `i64.mul_wide_u`, with low/high `i64` results. Add/subtract propagate carry/borrow; multiplication uses Metal's signed or unsigned `mulhi`. Tail calls reuse a shader call frame. Ordinary recursion is limited to 32 call frames by default; deeper calls trap. Larger call-depth limits can exceed the device’s stack capacity. Type-index blocks and calls can pass multiple values, including nullable and non-nullable typed function references. Recursive type groups and declared function, struct, and array subtyping are supported.

Each module can own or import multiple unshared linear memories with 32-bit or 64-bit addresses, active and passive data segments, globals, and a start function. Memories can use the default 65,536-byte page size or declare `(pagesize 1)` or `(pagesize 65536)`. Different page sizes can coexist and participate in `memory.copy`; imports must match the provider's page size. Supported memory instructions include loads, stores, `memory.size`, `memory.grow`, `memory.copy`, `memory.fill`, `memory.init`, and `data.drop`. Initializers support `i32` and `i64` add, subtract, multiply, and reads of earlier immutable globals. The default total memory limit is 16 MiB per module instance. Growth can add at most four pages per memory by default, subject to declared maxima and that total limit; excess growth returns `-1`. `memory.size`, `memory.grow`, and `KernelLimits::memory_growth_pages` count each memory's declared pages. `KernelLimits::memory_bytes` limits the sum of owned memory bytes per lane; it accepts byte limits up to 4 GiB without page alignment. Memory64 uses 64-bit addresses and page counts but retains these physical allocation bounds. `KernelLimits` lets callers raise the memory limit, growth reserve, table growth reserve, call depth, exception record limit, and GC heap size when compiling. Each growable memory reserves GPU buffer space for its possible growth, even when the total limit prevents every memory from growing in one lane. Larger reserves can reduce batch size. Large batches use consecutive dispatches sized for a 64 MiB buffer budget; per-lane state buffers are reused and reset between dispatches. A single invocation or shared buffer that exceeds a Metal buffer limit returns `Unsupported`. If an invocation traps, `run` and `run_values` return `Error::Trap` with its index in the original input and reason; results are discarded. `run_values_outcomes` returns a separate outcome for each invocation, as described below.

Each module can own or import multiple unshared function, external, exception, or GC reference tables, including typed and non-nullable element types, with 32-bit or 64-bit indices. Each table has at most 65,536 initial entries. Active, passive, and declared element segments are supported. `table.get`, `table.set`, `table.size`, `table.grow`, `table.fill`, `table.copy`, `table.init`, `elem.drop`, and `ref.null`, `ref.func`, `ref.is_null`, `ref.as_non_null`, `br_on_null`, `br_on_non_null`, `ref.test`, `ref.cast`, `br_on_cast`, and `br_on_cast_fail` operate on per-lane table state. Growth can add at most 256 entries by default, subject to the declared maximum and the 65,536-entry limit. Unchanged tables share an initialized buffer between lanes. `call_indirect` checks the slot and nominal function type at runtime. `call_ref` dispatches a typed function reference.

`Kernel::new_with_imports(wasm, export, providers)` links function, memory, table, global, and exception tag imports to exports from supplied Wasm provider modules. `providers` is a slice of `(namespace, wasm_bytes)` pairs. Each namespace has its own module state per lane; imported resources refer to the provider's state. Active segments and start functions run in dependency order. Imported functions run on the GPU. `Kernel::new_with_imports_and_limits` combines linking with custom limits. Linked module state must fit Metal's 31 buffer slots. Arbitrary host callbacks remain unsupported.

`Kernel::run_values_with_memory(inputs, memory_export, range)` returns function results and the requested byte range from an exported memory for each lane. The range must fit every lane's memory after execution. It can read a memory owned by the module or imported from a provider. Copying large ranges adds host transfer time.

`Kernel::run_values_with_memory_io(inputs, memory_export, input_offset, memory_inputs, output_range)` also copies a distinct byte slice into each lane's exported memory after data segments and start functions run, immediately before the exported call. Supply one equally sized byte slice per invocation. The input range must fit the memory's initial size. Use `0..0` as the output range when no memory readback is needed.

Metal has no native `f64`, and this Mac flushes native `f32` subnormals. Both types use GPU software arithmetic on raw bits. The `f64` primitives come from the vendored [metal-softfloat](https://github.com/guyfischman/metal-softfloat) source under the licenses in `licenses/`. Floating-point operations cost substantially more GPU work than native `i32` operations.

`Kernel::function_reference(export)` returns an opaque handle for a root-module function export. Pass it as `Value::FuncRef(Some(handle))`; use `Value::FuncRef(None)` for null. `run_values` also returns function handles. Handles belong to one compiled kernel and cannot cross kernel boundaries. Input checks enforce reference nullability and declared function subtyping.

`Kernel::external_reference(value)` retains a Rust value and returns an opaque `ExternalRef`. Pass it as `Value::ExternRef(Some(handle))`, or use `Value::ExternRef(None)` for null. `Kernel::external_reference_value::<T>(handle)` returns an `Arc<T>` to the retained value. The shader can store and pass its identity, but cannot read host data or call the host. References can pass through globals, tables, calls, and exception payloads. Tests and casts support all unshared function, external, exception, and GC heap types, including nullable bottom types. Concrete casts use nominal subtyping.

`Kernel::exception_reference(tag_export, payload)` creates an immutable exception for an exported tag. Pass it as `Value::ExnRef(Some(handle))`, or use `Value::ExnRef(None)` for null. `run_values` returns captured exception handles with their original tag and payload. `Kernel::exception_payload(handle)` reads raw payload values, including vectors and nested references. Returned handles can be passed to later invocations of the same kernel. Input serialization maps the reachable exception records into each lane and preserves shared child identities. Imported records count toward the lane's exception limit. Signatures with exception references also transfer an exception arena; numeric signatures retain their existing transfer format.

`Kernel::release_reference(value)` ends the active host lease for an external, GC, or exception reference and collects unreachable host records. All copied handles and external/any conversion aliases of that lease become invalid. Other retained snapshots and exception payloads keep reachable records alive. Reading a still-live exception payload renews the returned child leases; stale handles never become valid again. `Kernel::collect_garbage()` also reclaims unreachable records, including mixed GC/exception cycles and unpublished results from failed batches. Both methods require mutable access to the kernel. Dropping a copied handle alone does not release its lease. Null references and `i31` values need no storage; releasing them is a no-op. Scalars and non-null function references cannot be released. An `Arc<T>` already obtained through `external_reference_value` remains valid under normal Rust ownership. All reference handles belong to one compiled kernel; input and payload checks reject handles from other kernels. `Store` uses the same lease policy, as described below.


Exception handling supports tags, `try_table`, `throw`, `throw_ref`, `catch`, `catch_ref`, `catch_all`, and `catch_all_ref`. Calls propagate exceptions to the nearest matching handler. Tail calls remove the caller’s handlers. Traps cannot be caught. Exception references can be kept in locals, globals, or tables and rethrown. Each lane has capacity for 64 live immutable exception records by default; `KernelLimits::exception_records` changes this limit. Allocation reclaims unreachable records and mixed GC/exception cycles before reporting `exception record limit exceeded`. Live records retain their slot and payload. An escaping exception returns `Error::UncaughtException` with the original input index. Provider start exceptions stop initialization.

GC supports `anyref`, `eqref`, `i31ref`, structs, arrays, `ref.eq`, packed fields, all core struct and array instructions, and both external/any conversions. Arrays can initialize from passive data and element segments. GC constructors also work in constant expressions. Each lane has a nonmoving heap shared by its linked modules. Its default capacity is 1,024 64-bit words (8 KiB), including object headers. `KernelLimits::gc_heap_words` changes this bound. Allocation performs mark-sweep collection before reporting `GC resource limit exceeded`. Roots include live caller values, globals, tables, passive elements, and exception payloads. The collector handles cycles and coalesces adjacent free blocks. It does not compact live objects, so fragmentation can exhaust the heap. Larger heaps and caller root reserves reduce batch size.

`Value::I31(bits)` supplies the low 31 bits of an integer reference. `Value::GcRef(None)` supplies a null GC reference. `run_values` returns opaque `GcRef` handles for structs, arrays, and foreign host references. Passing a returned object handle into a later invocation copies its reachable graph into that lane; aliases and cycles are preserved. Returned mutable objects form new snapshots, and input snapshots stay unchanged. Each lane gets an independent copy. `Kernel::gc_reference_kind(handle)` identifies the abstract object kind. `Kernel::internalize_reference` and `Kernel::externalize_reference` implement host-side external/any conversions, including integer references and GC snapshots. `external_reference_value` reads only original Rust host values. Object fields remain accessible through Wasm instructions. GC objects and exception payloads can refer to each other.

Shared memory and types, continuation types, custom descriptors, and host interfaces such as WASI remain unsupported.

```rust
use metal_wasm::Kernel;

let wasm = std::fs::read("function.wasm")?;
let kernel = Kernel::new(&wasm, "calculate")?;
let output = kernel.run(&[1, 2, 3])?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Requires macOS 15 or later with a Metal device and the Rust toolchain. The backend uses `objc2-metal` 0.3.2 with retained Objective-C objects and typed Metal APIs. Shader compilation uses safe math and precise floating-point functions. Run `cargo test` to validate GPU execution. The kernel compiles once and can run repeatedly with different input batches. `run_timed` sums GPU execution time across dispatches; it excludes host-side buffer copies and submission.

## Persistent instances

`Store` keeps a fixed number of independent lanes alive across calls. Each lane
retains memories and their sizes, globals, tables and their sizes, segment drop
state, GC objects, and exception records. Different exports use the same state.
Instantiation runs the new module's initializers, active segments, and start
function once. Existing providers keep their state and do not restart.

```rust
use metal_wasm::Store;

let wasm = std::fs::read("function.wasm")?;
let mut store = Store::new(3)?;
let instance = store.instantiate(&wasm, &[])?;
let first = store.call_i32(instance, "calculate", &[1, 2, 3])?;
let second = store.call_i32(instance, "calculate", &[4, 5, 6])?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `Store::new_with_limits(lanes, limits)` to change resource bounds.
`call(instance, export, rows)` supports general signatures. Every call needs
exactly one input row per lane. `call_i32_timed` supplies the `(i32) -> i32`
interface and GPU command time. Pipelines compile on their first use; adding
an instance invalidates cached pipelines so indirect calls can reach new functions.

`instantiate(wasm, &[("provider", existing_instance)])` links to live provider
resources, including reexports and namespace aliases. Import limits are checked
against the provider's current memory and table sizes in every lane. Instance
and reference handles belong to one store. `function_reference(instance, export)`
and opaque external host values can be used in any lane of that store. Returned
GC and exception handles retain their live identity and can only be passed back
to their original lane. Public handles root their objects until explicitly released
or the store is dropped.
The matching reference conversion, kind, and exception payload methods are also
available on `Store`.

`get_global(instance, export)` reads one value per lane.
`set_global(instance, export, values)` writes one value per lane to a mutable global.
`read_memory(instance, export, range)` and
`write_memory(instance, export, offset, rows)` access current exported memories.
`table_size(instance, export)` returns each lane's current table size.
`read_table(instance, export, range)` reads one row of references per lane.
`write_table(instance, export, offset, rows)` writes one row per lane. Rows may
have different lengths; each must fit its lane's current table size.
Table ranges and offsets use `u64`, including for table64. All methods resolve
imported resource exports to their provider state. Global and table writes validate
the entire batch before mutation, including current per-lane bounds, types,
nullability, subtyping, reference ownership, and lane identity. Written references
become Wasm roots. Read references receive host leases.

Traps and uncaught exceptions retain changes made before execution stopped.
Other submitted lanes may also have completed. A failed start does not return an
instance handle; its resources remain internally retained because references
can have escaped into a provider. A failed Metal command makes the store unusable.

`Store::release_reference(value)` ends the active host lease for an external,
GC, or exception reference. All copied handles and external/any conversion aliases
of that lease become invalid. Wasm globals, tables, live element segments, GC
fields, and retained exception payloads keep reachable objects alive. Reading a
still-live object later returns a new valid handle; stale handles never become
valid again. Null references and `i31` values need no storage and release is a
no-op. Scalars and non-null function references cannot be released. Function identities
remain valid for the store's lifetime.

`Store::collect_garbage()` reclaims unreachable GC objects and exception records
and releases unreachable Rust host values. Collection also runs at dispatch
boundaries and after host reference release. It does not run start functions.
Dropping a copied handle alone does not release its lease. An `Arc<T>` obtained
through `external_reference_value` remains valid under normal Rust ownership.
Persistent resources must fit the device's Metal buffer limit. `Kernel` provides
fresh independent instances and GC snapshots for each invocation, with explicit
host reference release and collection.

## Per-lane outcomes

`Kernel::run_values_outcomes(inputs)` and
`Store::call_outcomes(instance, export, inputs)` return
`Result<Vec<LaneResult>, Error>`, where `LaneResult` is
`Result<Vec<Value>, LaneError>`. An outer error reports invalid inputs, resource
or serialization failures, or a failed Metal command. An inner error reports an
executed lane's Wasm trap or uncaught exception. Successful lanes retain their
results, including references. Later lanes and dispatches continue after a Wasm
failure; failed result rows are not decoded.

```rust
use metal_wasm::{LaneError, Value};
# fn report(kernel: &metal_wasm::Kernel) -> Result<(), metal_wasm::Error> {
for (lane, outcome) in kernel.run_values_outcomes(&[vec![Value::I32(42)]])?
    .into_iter().enumerate()
{
    match outcome {
        Ok(values) => println!("lane {lane}: {values:?}"),
        Err(LaneError::Trap { reason }) => println!("lane {lane}: {reason}"),
        Err(LaneError::UncaughtException) => println!("lane {lane}: uncaught exception"),
    }
}
# Ok(())
# }
```

Existing `run`, `run_values`, and `Store::call` methods retain their first-failure
error behavior. In a `Store`, writes made before a Wasm failure remain visible,
and other submitted lanes can also complete. Invalid input batches do not execute.

## Execution budgets

`KernelLimits::fuel` defaults to `Some(1_000_000)` per lane and dispatch. A function
entry and an executed loop iteration each consume one unit. Direct, indirect,
reference, and tail calls share the lane's budget across linked modules. In a
fresh `Kernel` invocation, starts and the exported call share the budget. In a
`Store`, instantiation and each call get a fresh budget. Exhaustion returns
`Error::Trap { input_index, reason: "fuel exhausted" }`; Wasm exception handlers
cannot catch it. Persistent writes before the trap survive.

Set `fuel: None` for trusted workloads that need no metering. Fuel bounds Wasm
control flow. It is not a wall-clock timeout or a count of Wasm instructions:
bulk operations and collection still depend on the configured resource reserves.

## Compatibility checks

Run `cargo run --example compat -- MODULE.wasm EXPORT I32_INPUT ...` with chosen inputs to check an exported `(i32) -> i32` function against fresh Wasmtime instances. It reports an unsupported module or signature with the reason, accepts matching traps and uncaught exceptions, and fails if Metal and Wasmtime disagree. For example, `cargo run --example compat -- tests/corpus/memory.wasm calculate 0 1 42` checks a compiled Rust memory workload. Use terminating functions: the GPU has a fuel budget, while this CPU comparison does not.

`cargo test --test compiled_corpus` checks eight compiled Rust, C, and C++ modules against Wasmtime and checks that a ninth compiled module reports its unsupported host import. The C and C++ fixtures include unoptimized and optimized builds. Tests cover numeric compiler lowering, SIMD, indirect calls, memory, and persistent C++ statics. `cargo test --test persistent_runtime` also compares a compiled Rust static with persistent Wasmtime instances and checks lane state, linking, traps, live references, and collection. Sources, binaries, and build commands are in `tests/corpus/`. `cargo test --test spec_runtime` runs all 258 scripts from the pinned official core tree, including scalar, control-flow, binary validation, SIMD and relaxed SIMD, references, exceptions, tail calls, multiple memories, memory64/table64, GC, linking, initialization, and bulk operations. No assertions are skipped. It uses actual persistent instances, including actions after traps, named invocations, registrations, global reads, and module definitions; see `tests/spec/README.md` for provenance and test resource limits. The harness also runs the complete wide arithmetic proposal script and four custom page size proposal scripts. These checks do not establish full Wasm conformance.

`cargo test --test generated_runtime` compares 24 reproducible generated programs
with Wasmtime across 2,352 fresh and persistent invocations. Seeds select arithmetic
sequences and direct, indirect, typed-reference, or tail calls. Programs also combine
GC chains, SIMD, `f64`, `i64`, two memories with different page sizes, bulk operations,
and traps or uncaught exceptions. Persistent checks compare returned values,
failure categories, globals, and memory after every call. This is a bounded,
deterministic matrix, not unrestricted fuzzing or proof of conformance.

`cargo test --test embedding_runtime --test kernel_references` checks atomic host
writes, provider aliases, typed tables, lane outcomes, reference release, stale
handles, retained graphs, and cyclic record reclamation.

## Performance comparison

Run `cargo run --release --example compare -- --wide-arithmetic` or
`--byte-pages` for the new feature workloads. Both also run in `--suite`.
`wide_arithmetic` executes 128 dependent rounds of signed and unsigned wide
multiplication, 128-bit addition, and subtraction. `byte_pages` uses a 261-byte
memory with one-byte pages, resets it on every call, and executes 128 dependent
rounds of byte loads, stores, and integer arithmetic. Both compare Metal,
Wasmtime, Wasmi, and optimized native Rust at 1, 1,000, 100,000, and 1,000,000
invocations. Each row reports the median of three samples and checks every
result. Metal reports both total and GPU command time. Compilation, CPU
instantiation, warmup, and result checks are outside timing.

Run `cargo run --release --example compare -- --lifetime` for repeated exception
and GC reclamation. Each lane makes 16 calls; each call creates, catches, and
discards 64 node/exception cycles. Metal uses an eight-record exception arena and
a 64-word heap, so it must reclaim within a call. Paired rows run the same code
with default fuel and with `fuel: None`. The benchmark checks every final digest
against Wasmtime and optimized native Rust. Wasmi lacks these features and is
omitted. Native Rust computes the same state updates with local values; it does
not allocate Wasm objects or run a collector. `count` means lanes. Compilation,
instantiation, reset, warmup, and verification are outside timing. Metal total
includes dispatch handling and collection between calls; GPU time includes
shader collection. The workload also runs in `--suite`.

Run `cargo run --release --example compare -- --persistent` to compare Metal,
Wasmtime, Wasmi, and native Rust with persistent state. At 1, 1,000, 4,096, and
10,000 lanes, each sample makes 16 calls per lane. Each call performs 24 rounds
of integer arithmetic using a mutable global and four memory cells. CPU Wasm
engines use one persistent instance per lane. The benchmark verifies every lane's
final digest and reports the median of three samples, including Metal total and
GPU command times. Compilation, instantiation, initialization, warmup, and digest
verification are outside the timed region. `count` means lanes; total calls are
`count * 16`. The workload is also included in `--suite`.

Run `cargo run --release --example compare` to compare Metal, Wasmtime, Wasmi, and optimized native Rust at 1, 1,000, 1,000,000, and 10,000,000 invocations. The example uses the same `x * 7 + 3` function for every path, checks every result, and prints median elapsed times in microseconds. It takes 31 samples of 100 runs for 1 invocation, 11 samples of 10 runs for 1,000, and 3 single-run samples for each larger count. Small runs are repeated inside each timed sample to reduce clock overhead.

Run `cargo run --release --example compare -- --metal-native-100m` to compare Metal and native Rust at 100,000,000 invocations. This mode takes 3 single-run samples and skips the Wasm engines.

Run `cargo run --release --example compare -- --mix` for a compute-heavy comparison at 1,000, 1,000,000, 10,000,000, and 100,000,000 invocations. Each input goes through 32 rounds of rotate, XOR, multiply, and add in Wasm on Metal, Wasmtime, Wasmi, and native Rust. The two CPU Wasm engines stop at 10,000,000 inputs. This mode reports Metal end-to-end time and Metal command-buffer GPU time separately. The GPU time comes from Metal's `GPUStartTime` and `GPUEndTime` after command completion; it excludes CPU buffer copies and submission. It includes GPU memory access and kernel execution. The benchmark fails if GPU timestamps are unavailable.

Run `cargo run --release --example compare -- --control` for the same counts with a 32-iteration Wasm loop and an `if`/`else` in every iteration. The `grouped` pattern places even inputs before odd inputs in groups of 32, so neighboring lanes usually take the same branch. The `mixed` pattern interleaves those same even and odd values, so neighboring lanes disagree. Both patterns execute the same arithmetic and number of iterations overall; only input order changes. Each path runs on Metal, Wasmtime, Wasmi, and native Rust, with the CPU Wasm engines omitted at 100,000,000 inputs. This mode uses three samples and checks every result. The `pattern` column identifies the input pattern; `metal_gpu_us` includes GPU buffer access and kernel execution.

Run `cargo run --release --example compare -- --suite` for fifteen workloads. `persistent_state` uses repeated calls against live per-lane state, as described above. `gc_graph` runs 64 rounds of struct and array allocation, mutation, and reclamation in each lane; run `cargo run --release --example compare -- --gc` to measure it alone. Its 2,560 allocated words per invocation exceed the default 1,024-word heap, so it exercises collection. It uses counts 1, 1,000, 10,000, and 100,000. The original `dispatch_i64`, `float_f64`, and `stateful_memory` cases remain. `bulk_start` uses a start function and bulk memory operations; `recursive_calls` uses variable-depth recursion; `mutable_table` uses passive elements, table updates, and indirect calls; `imported_calls` links a separate Wasm provider module; `linked_resources` imports and mutates provider memory, a global, and a table. `simd_i32x4` runs 32 rounds of four-lane integer arithmetic and a lane reduction. `reference_casts` runs 32 rounds of cast branches, typed dispatch, exception table capture and rethrow, and external-null tests. It measures GPU reference operations; public exception serialization and non-null external host values are covered by integration tests. `typed_exceptions` runs 32 rounds of typed function dispatch, divergent branches, and payload throws and catches. `tail_multi_resource` combines tail calls, extended constants, two memories, a 64-bit memory, and a 64-bit table. Run `cargo run --release --example compare -- --simd`, `--tail-multi-resource`, `--typed-exceptions`, or `--reference-casts` to measure one workload. Their definitions are in `examples/workloads/`. Each workload is checked against optimized native Rust and Wasmtime; Wasmi is omitted for the SIMD and new feature workloads because this version does not support their Wasm features. This mode reports the median of three single-run samples at 1, 1,000, 100,000, and 1,000,000 invocations, except the four memory workloads, which use 1, 1,000, 4,096, and 10,000 to exercise bounded dispatches.

The timed region includes result allocation and execution. Metal also includes per-lane state initialization, input transfer, command submission, and result transfer. Module compilation, CPU engine instantiation, warmup, input generation, and result verification are outside the timed region. Wasmtime and Wasmi reuse one instance per module and make one typed host call per input; the stateful and linked-resource workloads reset their touched state for each call. Metal batches one or more dispatches and runs each start function per lane. Thus `bulk_start` includes per-lane start work on Metal, while CPU start work occurs outside the timed region. These figures compare these APIs for these workloads, not peak engine or GPU throughput.

## Recorded benchmark results

The [wide arithmetic and byte-page results](examples/feature-benchmark-results.csv)
were measured on 2026-09-30 with an Apple M1 Max, 64 GB RAM, macOS 27.0,
and a release build with default fuel. All results matched Metal, Wasmtime,
Wasmi, and native Rust. These times are milliseconds at 1,000,000 invocations,
with 128 rounds per invocation and medians of three samples:

| Workload | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wide arithmetic | 14.38129 | 10.76975 | 257.20425 | 2056.88792 | 244.96158 |
| Byte pages | 59.69762 | 34.02750 | 557.26112 | 936.88908 | 364.66271 |

Metal total is about 17.9× faster than Wasmtime for wide arithmetic and 9.3×
for byte pages. The native path is one CPU thread. Metal total includes fresh
per-lane state, transfers, and submission; GPU time excludes host work.
The byte-page workload reserves 261 bytes per lane instead of a 64 KiB page.
These measurements describe these workloads and APIs, not general Wasm speed.
The CSV also includes 1, 1,000, and 100,000 invocations.

The [lifetime results](examples/lifetime-benchmark-results.csv) were measured on
2026-09-30 with an Apple M1 Max, 64 GB RAM, macOS 27.0, and a release build.
Each sample runs 16 calls per lane and 64 GC/exception cycles per call. These
times are milliseconds and are medians of three samples. Every final digest
matched Metal, Wasmtime, and native Rust. These rows use the default fuel budget.

| Lanes | Total calls | Metal total | Metal GPU | Wasmtime | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 10.32617 | 6.53829 | 0.17225 | 0.00175 |
| 1,000 | 16,000 | 16.84050 | 8.46363 | 169.70329 | 1.05858 |
| 4,096 | 65,536 | 32.95442 | 12.20779 | 702.01704 | 4.37975 |
| 10,000 | 160,000 | 68.31688 | 22.34583 | 1733.52433 | 10.81154 |

At 10,000 lanes, Metal total time is about 25.4× faster than Wasmtime. Native
Rust is about 6.3× faster than Metal total time; it uses local values without a
Wasm heap or collector. The CSV also includes paired runs with fuel disabled.
Three samples do not establish a precise metering cost. Metal GPU time includes
shader collection; total time also includes host collection between calls.

The earlier persistent and full-suite measurements below predate default fuel
and explicit reference release.

The [persistent results](examples/persistent-benchmark-results.csv) were measured
on 2026-09-30 with an Apple M1 Max, 64 GB RAM, macOS 27.0, and a release build.
Each sample executes 16 calls per lane. These times are milliseconds and are
medians of three samples; every final lane digest matched all four implementations.

| Lanes | Total calls | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 3.86221 | 0.49688 | 0.00167 | 0.00571 | 0.00125 |
| 1,000 | 16,000 | 6.48358 | 0.57817 | 3.25729 | 5.78158 | 0.79617 |
| 4,096 | 65,536 | 12.39679 | 0.88717 | 13.21992 | 23.68962 | 3.27104 |
| 10,000 | 160,000 | 23.75408 | 2.18216 | 32.09954 | 57.34012 | 7.86529 |

At 10,000 lanes, Metal total time is about 1.35× faster than Wasmtime and 2.41×
faster than Wasmi. Native Rust is about 3.02× faster than Metal total time.
Metal GPU time excludes host work and submission. The difference from total time
shows the cost of 16 synchronous calls through this API, including metadata,
argument, and result handling. This workload retains instance state on the GPU;
it does not copy each lane's full linear memory between calls.

The [full suite results](examples/benchmark-results.csv) were measured on 2026-09-30 with an Apple M1 Max, 64 GB RAM, macOS 27.0, and a release build. All thirteen workloads passed result verification. Each row reports the median of three samples. The new `gc_graph` workload produced these times in milliseconds:

| Invocations | Metal total | Metal GPU | Wasmtime | Native Rust |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.32462 | 6.89667 | 0.01804 | 0.00233 |
| 1,000 | 9.53346 | 7.49454 | 19.39467 | 2.23133 |
| 10,000 | 45.63967 | 25.38000 | 252.76796 | 22.24171 |
| 100,000 | 314.19962 | 205.30838 | 2278.65058 | 342.63042 |

Wasmtime 49.0.1 uses its `gc-drc` collector in this build, enabling reclamation during the allocation workload. Wasmi is omitted for this workload because it does not support GC. Native Rust uses equivalent array and struct computations with local values; it does not allocate a Wasm heap or run a collector. At 100,000 inputs, Metal total time is about 7.3× faster than Wasmtime and 1.1× faster than native Rust. At smaller counts, native Rust is faster. The bounded heap requires multiple Metal dispatches, and total time includes their state initialization and transfers. Metal GPU time includes heap reads, writes, allocation, and collection. The workload does not transfer GC graphs through the public reference API. These measurements describe this workload and API; they do not establish that GC operations in general run faster on a GPU.

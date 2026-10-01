# Benchmark overview — 2026-10-01

All five benchmark modes completed successfully. All returned results or final lane digests matched their native references. The dataset contains **89 result rows across 21 named workloads**. `control_flow` has two input patterns.

## Run and validation

- Code revision: `9914e72` (`Migrate the Metal backend to objc2-metal`).
- Host: Apple M1 Max, 64 GiB RAM, macOS 27.0 (26A428).
- Toolchain: Rust 1.98.0; release benchmark build.
- CPU engines: Wasmtime 49.0.1 and Wasmi 2.0.0. Wasmtime uses `gc-drc` for GC workloads.
- Metal binding: `objc2-metal` 0.3.2. Shader compilation uses safe math and precise floating-point functions.
- Full tests on this revision: **384 passed, zero failed**, including **58,001 official runtime cases** and **7,196 validation cases**. Metal API Validation was enabled.
- Doc-test check passed; the library has zero doc tests.
- Benchmarks ran sequentially after tests, with Metal API Validation disabled.

```sh
MTL_DEBUG_LAYER=1 cargo test --all-targets -- --test-threads=1 --nocapture
MTL_DEBUG_LAYER=1 cargo test --doc
cargo run --release --example compare -- --suite
cargo run --release --example compare
cargo run --release --example compare -- --metal-native-100m
cargo run --release --example compare -- --mix
cargo run --release --example compare -- --control
```

`--suite` includes persistent state and both lifetime variants. The individual feature flags select rows from that suite; they are not additional workloads.

## Reading the results

All tables below use **milliseconds**. The [complete CSV](examples/benchmark-results-2026-10-01.csv) preserves the runner's microsecond values and sample/repetition counts. The benchmark implementation is in [examples/compare.rs](examples/compare.rs).

- **Metal total** includes result allocation, per-lane state setup when applicable, transfers, command submission, synchronization, and result extraction. Persistent calls also include host collection between calls.
- **Metal GPU** is the sum of command-buffer execution times from `GPUStartTime` and `GPUEndTime`. It includes GPU memory access, shader execution, and shader collection. It excludes host work and submission.
- Total and GPU times are separate medians. Their difference estimates host/API costs; it is not an exact breakdown of individual copies or allocations.
- Compilation, CPU instantiation, warmup, input generation, and result verification are outside timing. Persistent reset/init/digest checks are also outside timing.
- Rows use medians of three samples, except simple arithmetic at one invocation (31 samples, 100 repetitions per sample) and 1,000 invocations (11 samples, 10 repetitions). Those repeated measurements are divided by the repetition count.
- `count` means independent invocations for stateless kernels. For persistent/lifetime workloads, it means lanes; each lane makes 16 timed calls. Multiply by 16 for total calls.
- CPU engines make one host call per input on one CPU thread. Stateless feature workloads reuse one CPU instance; stateful cases reset their touched state. Metal initializes independent per-lane instances. `bulk_start` includes per-lane starts on Metal; CPU starts occur during instantiation outside timing.
- Native Rust also uses one CPU thread and can benefit from compiler vectorization. It computes equivalent outputs with local values. In reference, exception, and GC cases it does not reproduce Wasm runtime checks, exception machinery, or collection. It is an optimized algorithm comparison.
- Metal uses the default per-dispatch fuel budget except `lifetime_cycles_unmetered`. CPU engines are not fuel-metered in these measurements.
- **—** means the runner does not measure that combination. Wasmi is omitted for five feature workloads and both lifetime variants because their required features are not supported by this benchmark's Wasmi path. CPU engines are omitted at 100 million invocations. Simple arithmetic does not report GPU-only time.
- Three samples and one host do not establish a precise metering cost, a causal branch-divergence effect, or general Wasm/GPU performance.

## Largest feature-suite counts

Each row below uses that workload's largest configured count. These counts differ between workloads. Compare engines within a row.

The last two columns are CPU time divided by **Metal total**. Values above 1 mean Metal total was faster; values below 1 mean the CPU path was faster.

| Workload | Count | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust | Wasmtime / Metal | Native / Metal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wide_arithmetic | 1,000,000 | 19.67225 | 11.99392 | 264.89267 | 2108.86229 | 254.32587 | 13.47× | 12.93× |
| byte_pages | 1,000,000 | 72.40904 | 40.16875 | 574.38288 | 975.15300 | 379.91012 | 7.93× | 5.25× |
| gc_graph | 100,000 | 276.36288 | 211.70904 | 1769.24508 | — | 216.63662 | 6.40× | 0.78× |
| reference_casts | 1,000,000 | 637.19138 | 88.70125 | 10565.72404 | — | 44.51538 | 16.58× | 0.0699× |
| typed_exceptions | 1,000,000 | 637.47883 | 67.06804 | 2576.08204 | — | 63.54271 | 4.04× | 0.0997× |
| tail_multi_resource | 10,000 | 54.62408 | 2.84654 | 0.70042 | — | 0.13846 | 0.0128× | 0.0025× |
| simd_i32x4 | 1,000,000 | 7.21462 | 0.66962 | 49.47625 | — | 26.97312 | 6.86× | 3.74× |
| dispatch_i64 | 1,000,000 | 10.77242 | 3.34200 | 80.55333 | 431.20875 | 56.21025 | 7.48× | 5.22× |
| float_f64 | 1,000,000 | 12.15667 | 5.70150 | 53.77500 | 210.22671 | 13.32125 | 4.42× | 1.10× |
| stateful_memory | 10,000 | 30.77600 | 0.55454 | 0.78342 | 4.89367 | 0.24050 | 0.0255× | 0.0078× |
| bulk_start | 10,000 | 27.72546 | 0.32371 | 0.61417 | 2.60396 | 0.19596 | 0.0222× | 0.0071× |
| recursive_calls | 1,000,000 | 12.41825 | 5.32733 | 44.50650 | 226.46512 | 10.33817 | 3.58× | 0.83× |
| mutable_table | 1,000,000 | 18.81583 | 1.61000 | 131.99771 | 489.12162 | 36.43054 | 7.02× | 1.94× |
| imported_calls | 1,000,000 | 7.83879 | 1.05100 | 88.56075 | 502.03633 | 73.99433 | 11.30× | 9.44× |
| linked_resources | 10,000 | 40.69558 | 0.55558 | 1.36233 | 5.14875 | 0.53067 | 0.0335× | 0.0130× |
| persistent_state | 10,000 | 29.24458 | 2.55421 | 50.71817 | 57.15504 | 8.74325 | 1.73× | 0.30× |
| lifetime_cycles | 10,000 | 120.58750 | 67.91271 | 1818.23421 | — | 11.35725 | 15.08× | 0.0942× |
| lifetime_cycles_unmetered | 10,000 | 124.21471 | 67.72658 | 1818.23421 | — | 11.35725 | 14.64× | 0.0914× |

## What this run shows

- Large integer workloads favor Metal in this run: wide arithmetic is 13.47× faster than Wasmtime and 12.93× faster than native Rust at one million invocations. Byte pages are 7.93× and 5.25× faster respectively.
- At 100 million invocations, 32-round integer mixing takes 710.83375 ms total on Metal versus 4169.85979 ms in native Rust (5.87× faster). Its GPU-only time is 17.42971 ms. The large total/GPU gap indicates substantial host/API costs; identifying individual causes needs profiling.
- For simple `x * 7 + 3`, native Rust wins at every measured count. At 100 million invocations, native takes 12.91479 ms versus 686.50662 ms on Metal (53.16× faster for native).
- Fresh stateful-memory, bulk-start, linked-resource, and tail/multi-resource workloads lose to the CPU engines at their largest counts. For example, stateful memory takes 30.77600 ms total, 0.55454 ms GPU-only, and 0.78342 ms in Wasmtime. The gap is consistent with large state-setup/host costs, but this measurement does not isolate them. CPU instance reuse and reset behavior also differ as described above.
- Persistent state at 10,000 lanes (160,000 calls) takes 29.24458 ms on Metal versus 50.71817 ms in Wasmtime and 8.74325 ms in native Rust. Lifetime cycles take 120.58750 ms versus 1818.23421 ms in Wasmtime; native local-value computation is 11.35725 ms.
- Reference casts beat Wasmtime by 16.58× at one million invocations, while native local-value computation is 14.31× faster than Metal total. Runtime-feature costs make this native comparison different from arithmetic-only cases.
- Mixed control flow did not show a GPU penalty in this run: at 100 million invocations, grouped GPU time is 40.15575 ms and mixed GPU time is 35.76208 ms. These medians do not establish that disagreement is free or that one input order is generally faster.

## Workload coverage

| Workload | Operations measured |
| --- | --- |
| affine | `x * 7 + 3`; basic API/transfer overhead at counts 1 through 100 million. |
| integer_mix | 32 unrolled dependent rounds of rotate, XOR, multiply, and add. |
| control_flow | A 32-iteration Wasm loop with if/else. Grouped and mixed use the same input multiset and arithmetic work, with different lane order. |
| wide_arithmetic | 128 dependent rounds of signed/unsigned wide multiplication and 128-bit addition/subtraction. |
| byte_pages | 128 rounds of byte loads, stores, and integer arithmetic in a 261-byte memory with one-byte pages. |
| gc_graph | 64 rounds of struct/array allocation, mutation, and reclamation in a bounded per-lane heap. |
| reference_casts | 32 rounds of reference cast branches, typed dispatch, captured exception rethrow, and external-null checks. |
| typed_exceptions | 32 rounds of typed function dispatch, divergent branches, and payload throws/catches. |
| tail_multi_resource | Tail calls, extended constants, multiple memories, memory64, and table64. |
| simd_i32x4 | 32 rounds of four-lane integer SIMD arithmetic and a lane reduction. |
| dispatch_i64 | 24 rounds of 64-bit arithmetic and indirect calls through four functions. |
| float_f64 | 12 rounds of double-precision multiply/divide/add with branches, then sqrt and saturating conversion. Metal emulates f64. |
| stateful_memory | A fresh per-invocation state with a mutable global, memory cells, and integer arithmetic. |
| bulk_start | Per-lane start work followed by bulk memory operations and arithmetic. |
| recursive_calls | Variable-depth recursive calls with integer arithmetic. |
| mutable_table | Passive elements, table mutation, indirect calls, and integer arithmetic. |
| imported_calls | Integer arithmetic through a separately linked Wasm provider. |
| linked_resources | Imported provider memory, global, and table state, mutated per invocation. |
| persistent_state | 16 calls per lane; each performs 24 arithmetic rounds using a global and four memory cells. |
| lifetime_cycles | 16 calls per lane; each creates/catches/discards 64 GC/exception cycles. Default fuel. |
| lifetime_cycles_unmetered | The same lifetime workload with Metal fuel disabled. |

## Every measured count

All engine times are milliseconds. Persistent/lifetime counts are lanes, with 16 calls per lane. The CSV records `calls_per_lane`, `samples`, and `repeats` explicitly.

### wide_arithmetic

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.33912 | 0.07154 | 0.00029 | 0.00208 | 0.00029 |
| 1,000 | 0.33567 | 0.07262 | 0.26850 | 2.13179 | 0.25046 |
| 100,000 | 2.98096 | 1.67288 | 26.89454 | 210.91917 | 25.00825 |
| 1,000,000 | 19.67225 | 11.99392 | 264.89267 | 2108.86229 | 254.32587 |

### byte_pages

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.33704 | 0.08558 | 0.00062 | 0.00104 | 0.00042 |
| 1,000 | 0.39496 | 0.08362 | 0.58646 | 0.95208 | 0.36742 |
| 100,000 | 15.40579 | 9.00483 | 56.86271 | 96.14588 | 37.93008 |
| 1,000,000 | 72.40904 | 40.16875 | 574.38288 | 975.15300 | 379.91012 |

### gc_graph

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.89600 | 4.53829 | 0.01425 | — | 0.00225 |
| 1,000 | 8.20879 | 6.28738 | 17.41000 | — | 2.11704 |
| 10,000 | 43.52025 | 24.36567 | 175.36025 | — | 21.73571 |
| 100,000 | 276.36288 | 211.70904 | 1769.24508 | — | 216.63662 |

### reference_casts

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.70821 | 0.45679 | 0.01008 | — | 0.00008 |
| 1,000 | 2.18900 | 0.56338 | 10.67367 | — | 0.04446 |
| 100,000 | 73.25604 | 10.12892 | 1052.30788 | — | 4.49454 |
| 1,000,000 | 637.19138 | 88.70125 | 10565.72404 | — | 44.51538 |

### typed_exceptions

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.41275 | 0.13408 | 0.00229 | — | 0.00012 |
| 1,000 | 2.33208 | 0.33462 | 2.54458 | — | 0.06213 |
| 100,000 | 73.84704 | 7.06321 | 258.49375 | — | 6.40217 |
| 1,000,000 | 637.47883 | 67.06804 | 2576.08204 | — | 63.54271 |

### tail_multi_resource

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.51279 | 0.11842 | 0.00012 | — | 0.00012 |
| 1,000 | 17.77217 | 0.26254 | 0.06742 | — | 0.01346 |
| 4,096 | 29.12133 | 1.13379 | 0.30367 | — | 0.05979 |
| 10,000 | 54.62408 | 2.84654 | 0.70042 | — | 0.13846 |

### simd_i32x4

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.39900 | 0.01550 | 0.00012 | — | 0.00008 |
| 1,000 | 0.28167 | 0.01571 | 0.05021 | — | 0.02704 |
| 100,000 | 1.42692 | 0.11550 | 5.01200 | — | 2.66267 |
| 1,000,000 | 7.21462 | 0.66962 | 49.47625 | — | 26.97312 |

### dispatch_i64

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.27429 | 0.04367 | 0.00012 | 0.00046 | 0.00008 |
| 1,000 | 0.30946 | 0.05808 | 0.08192 | 0.43696 | 0.05529 |
| 100,000 | 1.51375 | 0.40158 | 8.15308 | 43.06312 | 5.76100 |
| 1,000,000 | 10.77242 | 3.34200 | 80.55333 | 431.20875 | 56.21025 |

### float_f64

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.34592 | 0.13025 | 0.00012 | 0.00025 | 0.00008 |
| 1,000 | 0.38683 | 0.12867 | 0.05167 | 0.21108 | 0.01275 |
| 100,000 | 2.18683 | 1.16367 | 5.34871 | 21.02596 | 1.27583 |
| 1,000,000 | 12.15667 | 5.70150 | 53.77500 | 210.22671 | 13.32125 |

### stateful_memory

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.42754 | 0.04962 | 0.00017 | 0.00062 | 0.00004 |
| 1,000 | 9.63288 | 0.05779 | 0.07683 | 0.51792 | 0.02408 |
| 4,096 | 18.77096 | 0.28817 | 0.31688 | 2.01146 | 0.09854 |
| 10,000 | 30.77600 | 0.55454 | 0.78342 | 4.89367 | 0.24050 |

### bulk_start

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.26188 | 0.02746 | 0.00008 | 0.00033 | 0.00004 |
| 1,000 | 8.71454 | 0.03825 | 0.06050 | 0.24496 | 0.01908 |
| 4,096 | 16.58612 | 0.15579 | 0.25992 | 0.98717 | 0.07721 |
| 10,000 | 27.72546 | 0.32371 | 0.61417 | 2.60396 | 0.19596 |

### recursive_calls

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.28871 | 0.05104 | 0.00008 | 0.00025 | 0.00004 |
| 1,000 | 0.43833 | 0.09262 | 0.04479 | 0.22529 | 0.01029 |
| 100,000 | 2.19462 | 1.18267 | 4.67062 | 22.39233 | 0.99321 |
| 1,000,000 | 12.41825 | 5.32733 | 44.50650 | 226.46512 | 10.33817 |

### mutable_table

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.30029 | 0.03392 | 0.00038 | 0.00054 | 0.00008 |
| 1,000 | 0.35812 | 0.04112 | 0.13500 | 0.49250 | 0.03704 |
| 100,000 | 2.34167 | 0.22054 | 16.12104 | 90.07471 | 3.75312 |
| 1,000,000 | 18.81583 | 1.61000 | 131.99771 | 489.12162 | 36.43054 |

### imported_calls

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.27383 | 0.01817 | 0.00017 | 0.00054 | 0.00017 |
| 1,000 | 0.30179 | 0.01858 | 0.09592 | 0.48608 | 0.07104 |
| 100,000 | 1.36096 | 0.14271 | 8.72475 | 50.56383 | 7.38067 |
| 1,000,000 | 7.83879 | 1.05100 | 88.56075 | 502.03633 | 73.99433 |

### linked_resources

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.32125 | 0.05058 | 0.00029 | 0.00062 | 0.00008 |
| 1,000 | 14.46804 | 0.05788 | 0.13117 | 0.51817 | 0.05488 |
| 4,096 | 22.94121 | 0.28179 | 0.53717 | 2.12271 | 0.20883 |
| 10,000 | 40.69558 | 0.55558 | 1.36233 | 5.14875 | 0.53067 |

### persistent_state

| Lanes | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.88158 | 0.47425 | 0.00171 | 0.00571 | 0.00129 |
| 1,000 | 7.65908 | 0.56971 | 3.24050 | 6.07050 | 0.90325 |
| 4,096 | 14.72592 | 1.11504 | 13.87904 | 23.62642 | 3.27704 |
| 10,000 | 29.24458 | 2.55421 | 50.71817 | 57.15504 | 8.74325 |

### lifetime_cycles

| Lanes | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 21.21767 | 16.00929 | 0.17362 | — | 0.00171 |
| 1,000 | 38.79950 | 27.50938 | 179.15117 | — | 1.06767 |
| 4,096 | 43.00025 | 19.30742 | 726.29400 | — | 4.54021 |
| 10,000 | 120.58750 | 67.91271 | 1818.23421 | — | 11.35725 |

### lifetime_cycles_unmetered

| Lanes | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.01288 | 19.72588 | 0.17362 | — | 0.00171 |
| 1,000 | 37.60825 | 25.88400 | 179.15117 | — | 1.06767 |
| 4,096 | 41.23750 | 17.24938 | 726.29400 | — | 4.54021 |
| 10,000 | 124.21471 | 67.72658 | 1818.23421 | — | 11.35725 |

### affine

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.24803 | — | 0.00003 | 0.00006 | 0.00001 |
| 1,000 | 0.25025 | — | 0.02352 | 0.04634 | 0.00008 |
| 1,000,000 | 7.49017 | — | 22.87142 | 45.50633 | 0.09225 |
| 10,000,000 | 67.12108 | — | 234.28450 | 456.20708 | 1.28017 |
| 100,000,000 | 686.50662 | — | — | — | 12.91479 |

### integer_mix — straight

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.32592 | 0.01208 | 0.03725 | 0.15317 | 0.02342 |
| 1,000,000 | 7.48958 | 0.19412 | 36.71392 | 154.55292 | 23.11675 |
| 10,000,000 | 68.07196 | 1.73725 | 371.32971 | 1544.91517 | 231.35958 |
| 100,000,000 | 710.83375 | 17.42971 | — | — | 4169.85979 |

### control_flow — grouped

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1.78792 | 0.01742 | 0.04321 | 0.91571 | 0.02379 |
| 1,000,000 | 10.65208 | 0.39038 | 72.65650 | 279.89171 | 44.38429 |
| 10,000,000 | 136.90283 | 4.67483 | 543.62738 | 3004.96404 | 280.28683 |
| 100,000,000 | 714.25062 | 40.15575 | — | — | 2335.35821 |

### control_flow — mixed

| Invocations | Metal total | Metal GPU | Wasmtime | Wasmi | Native Rust |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1.99958 | 0.01738 | 0.12175 | 0.49642 | 0.05833 |
| 1,000,000 | 7.57475 | 0.39054 | 86.76488 | 384.05275 | 28.71333 |
| 10,000,000 | 81.72725 | 3.62646 | 577.87492 | 2980.96258 | 298.91817 |
| 100,000,000 | 688.47046 | 35.76208 | — | — | 2318.29729 |

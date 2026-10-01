# Historical benchmark results

These measurements predate the latest [benchmark overview](../BENCHMARKS.md).


The [wide arithmetic and byte-page results](feature-benchmark-results.csv)
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

The [lifetime results](lifetime-benchmark-results.csv) were measured on
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

The [persistent results](persistent-benchmark-results.csv) were measured
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

The [full suite results](benchmark-results.csv) were measured on 2026-09-30 with an Apple M1 Max, 64 GB RAM, macOS 27.0, and a release build. All thirteen workloads passed result verification. Each row reports the median of three samples. The new `gc_graph` workload produced these times in milliseconds:

| Invocations | Metal total | Metal GPU | Wasmtime | Native Rust |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.32462 | 6.89667 | 0.01804 | 0.00233 |
| 1,000 | 9.53346 | 7.49454 | 19.39467 | 2.23133 |
| 10,000 | 45.63967 | 25.38000 | 252.76796 | 22.24171 |
| 100,000 | 314.19962 | 205.30838 | 2278.65058 | 342.63042 |

Wasmtime 49.0.1 uses its `gc-drc` collector in this build, enabling reclamation during the allocation workload. Wasmi is omitted for this workload because it does not support GC. Native Rust uses equivalent array and struct computations with local values; it does not allocate a Wasm heap or run a collector. At 100,000 inputs, Metal total time is about 7.3× faster than Wasmtime and 1.1× faster than native Rust. At smaller counts, native Rust is faster. The bounded heap requires multiple Metal dispatches, and total time includes their state initialization and transfers. Metal GPU time includes heap reads, writes, allocation, and collection. The workload does not transfer GC graphs through the public reference API. These measurements describe this workload and API; they do not establish that GC operations in general run faster on a GPU.

# WebAssembly spec runtime cases

All 258 `.wast` scripts in the pinned [official core tree](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core) are included, byte-for-byte, at commit `608711107b7f1edb13efd57b7d79b49477462d36`.
They are covered by [`licenses/WebAssembly-spec-Apache-2.0.txt`](../../licenses/WebAssembly-spec-Apache-2.0.txt).

The tree contains 97 root scripts, 59 SIMD scripts, 41 multi-memory scripts,
25 memory64 scripts, 17 GC scripts, eight bulk-memory scripts, seven relaxed
SIMD scripts, and four exception scripts. Earlier fixtures retain their flat
names. The remaining fixtures are under `core/` with their upstream paths.
Flat prefixes identify the upstream subdirectory: `simd_`, `multi_memory_`,
`memory64_`, `gc_`, `bulk_memory_`, and `exceptions_`. The renamed root scripts
are `select_functions.wast` (`select.wast`) and `table_grow_instances.wast`
(`table_grow.wast`). `extended_const_elements.wast` retains the earlier exact
lines 1055–1110 of `elem.wast`; the complete `elem.wast` also runs.

Run the suite on a Mac with Metal:

```sh
cargo test --test spec_runtime -- --test-threads=1 --nocapture
```

Every directive is processed. No assertions are filtered or skipped. The harness
checks return values, traps, exceptions, exhaustion, unlinkable modules, and
malformed and invalid module rejection. Float results use exact bits or the
specified NaN patterns. Relaxed SIMD accepts the alternatives permitted by the
script. Rejection assertions check the error category, not diagnostic wording.
Directives are parsed separately because some invalid modules cannot be encoded
by the text parser. Valid Unicode names, including bidi characters, are allowed.

Every script runs in one persistent `Store` lane through the public runtime.
Actions execute once; later assertions see writes made before traps and failed
starts. Named instances, registrations, imports, global reads, and module
instances retain their real state. `ref.extern N` and `ref.host N` share one
retained Rust host identity. Public reference methods check returned identities
and kinds. A `module definition` validates and retains bytes without allocating
its potentially enormous memory or table. A `module instance` instantiates it.
The pure Wasm `spectest` provider supplies the specified scalar print functions,
immutable globals, tables, and memory.

Most scripts reserve 32 call frames, eight growth pages, 65,536 table growth
entries, 512 exception records, 4,096 GC words, and two million fuel units.
`call.wast`, `call_indirect.wast`, and `call_ref.wast` reserve 256 frames.
The first two also reserve 32 MiB and 512 growth pages for a 306-page grow.
`local_tee.wast` and `nop.wast` reserve 64 growth pages; multi-memory
`memory_grow.wast` reserves 16. Both root and memory64 `memory_grow.wast`
reserve 64 MiB and 1,024 growth pages for their successful growth to 804 pages.
Production defaults remain unchanged.

This is complete script coverage for this pinned core tree, not a formal proof
of conformance or coverage of every proposal and host embedding. Threads and
host interfaces such as WASI remain outside the supported GPU profile. Focused
integration tests also cover resource bounds, multiple lanes, public reference
transport, reclamation, and execution budgets beyond the official scripts.

## Proposal tests

`wide-arithmetic.wast` comes unchanged from the [official wide arithmetic tree](https://github.com/WebAssembly/wide-arithmetic/blob/024d4ce1b28cee7fd006d9c5175e3b7ece52b429/test/core/wide-arithmetic.wast),
pinned at `024d4ce1b28cee7fd006d9c5175e3b7ece52b429`. Its 99 runtime and ten
module/validation cases all run. `wide_arithmetic.rs` also compares 10,368
inputs across Metal lanes with Rust `u128`/`i128` and Wasmtime, including carry,
borrow, wraparound, signed extrema, and deterministic random inputs. The same
WebAssembly Apache-2.0 license applies.

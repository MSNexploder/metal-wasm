# WebAssembly spec runtime cases

The eight scalar scripts come from [WebAssembly spec core tests](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core). The `simd_*.wast` scripts come from [core SIMD tests](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core/simd). The six relaxed SIMD scripts come from [core relaxed SIMD tests](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core/relaxed-simd). All are pinned at commit `608711107b7f1edb13efd57b7d79b49477462d36` and covered by
[`licenses/WebAssembly-spec-Apache-2.0.txt`](../../licenses/WebAssembly-spec-Apache-2.0.txt).

The added `const.wast`, `return_call*.wast`, `multi_memory_*.wast`, and `memory64_*.wast` files come from the same pinned core test tree. `extended_const_elements.wast` contains exact lines 1055–1110 of that tree's `elem.wast`. The runtime harness supplies a pure Wasm `spectest` provider for the print function and immutable global used by the added scripts.

`spec_runtime.rs` checks return and trap assertions, malformed and invalid module rejection, exact float bits, NaN patterns, and allowed alternatives for relaxed SIMD. Validation cases assert rejection without matching diagnostic text. Directives are parsed separately so the invalid multiple-supertype declaration in `gc_type-subtyping.wast` can be rejected by the text parser without preventing later assertions from running. The tail-call scripts now include their public null function-reference assertions. No assertions are skipped in the included scripts.

Each invocation gets fresh state by default. Reference and table scripts run in replay mode: the harness emits a wrapper that replays preceding successful actions within one lane before the current assertion. It supplies opaque Rust-backed external values for `ref.extern N`, shares one handle for each identity within the wrapper, and verifies returned identities by retrieving their host values. Wrapper signatures preserve concrete reference types and null arguments. `ref.host N` uses the same opaque host identity as `ref.extern N`, with the Wasm external-to-any conversion in replay wrappers. The harness checks returned GC reference kinds and externalized host identities through the public API. Trapped or exceptional actions are not replayed. This preserves the state needed by subsequent included assertions, but does not independently verify that failed bulk operations left their destination unchanged. The harness validates `module definition` without allocating its potentially enormous declared table. Production defaults remain unchanged: runtime scripts reserve 256 call frames and 65,536 table growth entries.

`br_on_null.wast`, `br_on_non_null.wast`, `call_ref.wast`, `ref_as_non_null.wast`, `ref_func.wast`, `return_call_ref.wast`, and `type-equivalence.wast` come unchanged from the pinned core tree. `type-rec.wast` is the complete pinned script, including mixed recursive groups, GC types, and their linking checks. `exceptions_{tag,throw,throw_ref,try_table}.wast` come unchanged from `test/core/exceptions/`. The harness checks tag linking, `assert_unlinkable`, and `assert_exception`, including imported tag aliases. Integration tests cover public exception serialization, nested payload aliases, vectors, exact float bits, external payload values, provider tables, and resource limits.

The following new scripts come unchanged from the same pin:

- `select_functions.wast` is now the complete core `select.wast`, including its three external-reference assertions.
- `local_init.wast`, `table.wast`, `table_get.wast`, `table_set.wast`, and `table_size.wast` come from `test/core/`.
- `bulk_memory_table_fill.wast` comes from `test/core/bulk-memory/table_fill.wast`.
- `memory64_table64.wast` and `memory64_table_{get,set,size,grow,fill}64.wast` come from `test/core/memory64/`.

All 17 `gc_*.wast` files come unchanged from the pinned [`test/core/gc/`](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core/gc) directory. They cover i31 values, reference equality and conversions, structs, arrays and bulk array operations, casts and cast branches, subtyping, and binary rejection. `ref_null.wast` is also complete. These replace the previous function-only recursive-type and non-GC reference subsets. Together with `type-rec.wast`, these 19 files contain 606 runtime assertions, 99 rejection assertions, and 108 valid module declarations. Every directive in each included script is processed; none is filtered or skipped.

`table_grow_instances.wast` contains core `table_grow.wast` lines 1–111 and 133–198. Lines 112–132 grow a persistent provider instance and then link a new module against its changed size. That persistent linking model remains outside this runner.

These checks do not establish full Wasm conformance. Focused integration tests cover resource limits, per-lane isolation, public reference transport, and garbage collection behavior that the pinned scripts do not test.

`simd_align.wast` and `simd_const.wast` require persistent state and remain excluded. `simd_linking.wast` and `simd_memory-multi.wast` contain no runtime assertions for this harness. Other scripts requiring persistent linking or unsupported host interfaces remain outside the harness.

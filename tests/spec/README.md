# WebAssembly spec runtime cases

The eight scalar scripts come from [WebAssembly spec core tests](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core). The `simd_*.wast` scripts come from [core SIMD tests](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core/simd). The six relaxed SIMD scripts come from [core relaxed SIMD tests](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core/relaxed-simd). All are pinned at commit `608711107b7f1edb13efd57b7d79b49477462d36` and covered by
[`licenses/WebAssembly-spec-Apache-2.0.txt`](../../licenses/WebAssembly-spec-Apache-2.0.txt).

The added `const.wast`, `return_call*.wast`, `multi_memory_*.wast`, and `memory64_*.wast` files come from the same pinned core test tree. `extended_const_elements.wast` contains exact lines 1055–1110 of that tree's `elem.wast`. The runtime harness supplies a pure Wasm `spectest` provider for the print function and immutable global used by the added scripts.

`spec_runtime.rs` checks return and trap assertions, malformed and invalid module rejection, exact float bits, NaN patterns, and allowed alternatives for relaxed SIMD. Validation cases assert rejection without matching diagnostic text. Directives are parsed separately so the invalid multiple-supertype declaration in `gc_type-subtyping.wast` can be rejected by the text parser without preventing later assertions from running. The tail-call scripts now include their public null function-reference assertions. No assertions are skipped in the included scripts.

Every script runs in one persistent `Store` lane. Invocations, global reads,
named instances, registrations, and instantiation use the actual public runtime.
Actions are executed once. Later assertions observe state after both successful
and trapping actions, including failed bulk operations and failed starts.
`ref.extern N` and `ref.host N` share one retained Rust host identity throughout
the script. Reference results are checked through the public kind and conversion
methods. `module definition` validates and retains bytes without allocating its
potentially enormous table; `module instance` instantiates a retained definition.
The Wasm `spectest` provider supplies scalar print functions, immutable globals,
tables, and memory. Production defaults remain unchanged: most scripts reserve
32 call frames, eight growth pages, 65,536 table growth entries, 512 exception
records, and 4,096 GC words. The complete `memory_grow.wast` script uses a 64 MiB
memory bound and 1,024 growth pages to cover its successful growth to 804 pages.
`call_ref.wast` uses 256 frames for its 200-deep recursion assertions.

`br_on_null.wast`, `br_on_non_null.wast`, `call_ref.wast`, `ref_as_non_null.wast`, `ref_func.wast`, `return_call_ref.wast`, and `type-equivalence.wast` come unchanged from the pinned core tree. `type-rec.wast` is the complete pinned script, including mixed recursive groups, GC types, and their linking checks. `exceptions_{tag,throw,throw_ref,try_table}.wast` come unchanged from `test/core/exceptions/`. The harness checks tag linking, `assert_unlinkable`, and `assert_exception`, including imported tag aliases. Integration tests cover public exception serialization, nested payload aliases, vectors, exact float bits, external payload values, provider tables, and resource limits.

The following new scripts come unchanged from the same pin:

- `select_functions.wast` is now the complete core `select.wast`, including its three external-reference assertions.
- `local_init.wast`, `table.wast`, `table_get.wast`, `table_set.wast`, and `table_size.wast` come from `test/core/`.
- `bulk_memory_table_fill.wast` comes from `test/core/bulk-memory/table_fill.wast`.
- `memory64_table64.wast` and `memory64_table_{get,set,size,grow,fill}64.wast` come from `test/core/memory64/`.

All 17 `gc_*.wast` files come unchanged from the pinned [`test/core/gc/`](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core/gc) directory. They cover i31 values, reference equality and conversions, structs, arrays and bulk array operations, casts and cast branches, subtyping, and binary rejection. `ref_null.wast` is also complete. These replace the previous function-only recursive-type and non-GC reference subsets. Together with `type-rec.wast`, these 19 files contain 606 runtime assertions, 99 rejection assertions, and 108 valid module declarations. Every directive in each included script is processed; none is filtered or skipped.

`table_grow_instances.wast` is now the complete core `table_grow.wast`, including
linking a new module against an already grown provider table.

The complete `data`, `elem`, `global`, `imports`, `linking`, `start`, `instance`,
`memory_size`, `memory_grow`, `memory_redundancy`, and `memory_trap` scripts come
unchanged from the same pin. All eight `test/core/bulk-memory/*.wast` files are
included. `simd_align.wast`, `simd_const.wast`, `simd_linking.wast`, and
`simd_memory-multi.wast` are also complete pinned files. Every directive in each
included script is processed; no assertions are skipped.

These checks do not establish full Wasm conformance. Focused integration tests cover resource limits, per-lane isolation, public reference transport, and garbage collection behavior that the pinned scripts do not test.

Other unvendored core scripts and tests requiring unsupported host interfaces or
threading remain outside this harness. Whole-script coverage is recorded only for
the included files; it is not a claim that the full official suite passes.

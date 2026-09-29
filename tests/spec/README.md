# WebAssembly spec runtime cases

The eight scalar scripts come from [WebAssembly spec core tests](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core). The `simd_*.wast` scripts come from [core SIMD tests](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core/simd). The six relaxed SIMD scripts come from [core relaxed SIMD tests](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core/relaxed-simd). All are pinned at commit `608711107b7f1edb13efd57b7d79b49477462d36` and covered by
[`licenses/WebAssembly-spec-Apache-2.0.txt`](../../licenses/WebAssembly-spec-Apache-2.0.txt).

`spec_runtime.rs` runs all 35,406 return and trap assertions and checks rejection of all 1,248 malformed and invalid modules in the 69 included scripts. It checks exact float bits, canonical and arithmetic NaN patterns, and allowed alternatives for relaxed SIMD. Validation cases assert rejection, without matching the reference interpreter's diagnostic text.

Each invocation gets fresh state. The `simd_align.wast` and `simd_const.wast` scripts contain assertions that depend on memory or globals persisting across separate invocations, so they are excluded as whole scripts. `simd_linking.wast` and `simd_memory-multi.wast` contain no runtime assertions for this harness. Individual vector calls, globals, memory access, and mixed scalar/vector signatures are compared with Wasmtime in the library tests.

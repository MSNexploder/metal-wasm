# WebAssembly spec runtime cases

`i32.wast`, `i64.wast`, `f32_cmp.wast`, `f64_cmp.wast`, `conversions.wast`,
`f32.wast`, `f64.wast`, and `if.wast` come from the [WebAssembly specification test suite](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core).
They are pinned at commit `608711107b7f1edb13efd57b7d79b49477462d36` and covered by
[`licenses/WebAssembly-spec-Apache-2.0.txt`](../../licenses/WebAssembly-spec-Apache-2.0.txt).

`spec_runtime.rs` runs all 11,275 return and trap assertions in these scripts.
It checks exact float bits and the specification's canonical and arithmetic NaN
patterns. It counts and excludes 295 malformed and invalid module assertions
because those test validation, while this harness checks execution of supported
exports. Each invocation gets fresh state, so scripts that require persistent
state across calls are outside this harness.

# WebAssembly spec runtime cases

`i32.wast` and `i64.wast` come from the [WebAssembly specification test suite](https://github.com/WebAssembly/spec/tree/608711107b7f1edb13efd57b7d79b49477462d36/test/core).
They are pinned at commit `608711107b7f1edb13efd57b7d79b49477462d36` and covered by
[`licenses/WebAssembly-spec-Apache-2.0.txt`](../../licenses/WebAssembly-spec-Apache-2.0.txt).

`spec_runtime.rs` runs all return and trap assertions in these two scripts.
It counts and excludes malformed and invalid module assertions because those
test validation, while this harness checks execution of supported exports.

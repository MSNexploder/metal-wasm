# Development and release checks

## Environment

Use a Mac running macOS 15 or later with a Metal GPU, Rust 1.98 or later,
and Xcode command-line tools. The test suite executes real Metal commands.
A machine without a Metal device cannot provide runtime validation.
The checked-in compiler corpus needs no Wasm compiler to run.

## Validate a candidate

Run the complete local gate from any working directory:

```sh
./scripts/check-release.sh
```

The script stops at the first failure. It checks the current candidate,
including uncommitted changes, and does not commit or publish it. It retains
Cargo artifacts in `CARGO_TARGET_DIR` or `target/`. It removes its temporary
package extraction when it exits.

The gate runs these checks in order:

1. `cargo fmt --check`.
2. `cargo clippy --locked --all-targets -- -D warnings`.
3. `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps`.
4. `MTL_DEBUG_LAYER=1 cargo test --locked --all-targets -- --test-threads=1 --nocapture`.
5. `MTL_DEBUG_LAYER=1 cargo test --locked --doc`.
6. `cargo package --locked --allow-dirty`, including Cargo's package build verification.
7. `cargo check --locked --all-targets` against the extracted `.crate` archive.

`missing_docs` is enabled for the library. Rustdoc compiles the public API
examples; file-loading examples are `no_run` because their module bytes
come from the caller. The runtime tests cover actual GPU execution.

For a localized change, run the affected integration target first:

```sh
MTL_DEBUG_LAYER=1 cargo test --locked --test execution_limits -- --test-threads=1
```

The [official suite](tests/spec/README.md) records the exact pinned upstream tree
and assertion policy. Keep its fixtures unchanged. Add regressions at public
runtime boundaries and compare numeric behavior with an independent CPU engine.
The [compiled corpus](tests/corpus/README.md) records source and rebuild commands.
Do not regenerate binary fixtures merely to make a test pass.

## Benchmarks

Run benchmarks sequentially after tests, with `MTL_DEBUG_LAYER` unset. The
[benchmark overview](BENCHMARKS.md) lists all five commands, timing boundaries,
workload definitions, and the full recorded results. Keep raw CSV results with
the code revision, toolchain, host, counts, and sample settings used for the run.
Do not compare a GPU-only time with an end-to-end CPU time as an API speedup.

## Before publication

- Pass the complete local gate and review the changes.
- Commit the validated candidate and confirm that `git status --short` is empty.
- Check the package name, version, description, and minimum Rust version.
- Retain `LICENSE-MIT`, `LICENSE-APACHE`, and all third-party notices in `licenses/`.
- Run `cargo publish --dry-run --locked` from the clean candidate checkout.
- Configure and verify the canonical remote before publishing source or a release tag.

Publication requires maintainer authorization. The local gate does not upload
source, create a release tag, or publish to crates.io. Documentation builds use
an Apple target through [docs.rs metadata](https://docs.rs/about/metadata).
The supported bounded GPU profile in the README remains the release contract;
official script coverage does not imply support for WASI or shared memory.

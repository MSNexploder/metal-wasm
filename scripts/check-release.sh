#!/bin/sh
# Validate a local release candidate. This does not publish anything.
set -eu

if [ "$(uname -s)" != Darwin ]; then
    echo "Release checks require macOS 15 or later and a Metal GPU." >&2
    exit 1
fi
release_macos=$(sw_vers -productVersion)
if [ "${release_macos%%.*}" -lt 15 ]; then
    echo "Release checks require macOS 15 or later." >&2
    exit 1
fi

release_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$release_root"
release_target=${CARGO_TARGET_DIR:-"$release_root/target"}
case "$release_target" in
    /*) ;;
    *) release_target="$release_root/$release_target" ;;
esac
export CARGO_TARGET_DIR="$release_target"

cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings" cargo doc --locked --no-deps
MTL_DEBUG_LAYER=1 cargo test --locked --all-targets -- --test-threads=1 --nocapture
MTL_DEBUG_LAYER=1 cargo test --locked --doc
# Allow uncommitted candidate changes so this gate can run before committing.
cargo package --locked --allow-dirty

# Compile all packaged targets to catch omitted fixtures and example sources.
release_id=$(cargo pkgid --locked)
release_version=${release_id##*#}
release_version=${release_version##*@}
release_tmp=$(mktemp -d "${TMPDIR:-/tmp}/metal-wasm-release.XXXXXX")
trap 'rm -rf "$release_tmp"' EXIT
trap 'exit 1' HUP INT TERM
tar -xzf "$release_target/package/metal-wasm-$release_version.crate" -C "$release_tmp"
cargo check --locked --all-targets \
    --manifest-path "$release_tmp/metal-wasm-$release_version/Cargo.toml"
echo "Release checks passed. No package was published."

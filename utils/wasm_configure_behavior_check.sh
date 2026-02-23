#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${1:-utils/configure_wasm_build.sh}"

if [[ ! -x "$SCRIPT" ]]; then
  echo "[wasm-config-behavior] missing executable: $SCRIPT" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

fake_bin="$tmpdir/fake-bin"
build_dir="$tmpdir/build"
missing_src="$tmpdir/missing-llvm-src"
mkdir -p "$fake_bin" "$build_dir"

cat >"$fake_bin/emcmake" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
touch "${WASM_CONFIG_EMCMAKE_MARKER:?}"
exec "$@"
EOF
chmod +x "$fake_bin/emcmake"

cat >"$fake_bin/cmake" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
touch "${WASM_CONFIG_CMAKE_MARKER:?}"
exit 0
EOF
chmod +x "$fake_bin/cmake"

emcmake_marker="$tmpdir/emcmake.called"
cmake_marker="$tmpdir/cmake.called"

# print mode should not require source directory existence.
BUILD_DIR="$build_dir" LLVM_SRC_DIR="$missing_src" "$SCRIPT" --print-cmake-command >/dev/null

set +e
PATH="$fake_bin:$PATH" \
WASM_CONFIG_EMCMAKE_MARKER="$emcmake_marker" \
WASM_CONFIG_CMAKE_MARKER="$cmake_marker" \
BUILD_DIR="$build_dir" \
LLVM_SRC_DIR="$missing_src" \
EMCMAKE_BIN=emcmake \
CMAKE_BIN=cmake \
"$SCRIPT" >/tmp/wasm-config-behavior.out 2>/tmp/wasm-config-behavior.err
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  echo "[wasm-config-behavior] configure unexpectedly succeeded with missing LLVM source dir" >&2
  exit 1
fi

if ! grep -Fq -- "missing LLVM source directory" /tmp/wasm-config-behavior.err; then
  echo "[wasm-config-behavior] missing explicit LLVM source directory diagnostic" >&2
  cat /tmp/wasm-config-behavior.err >&2
  exit 1
fi

if [[ -e "$emcmake_marker" || -e "$cmake_marker" ]]; then
  echo "[wasm-config-behavior] configure invoked emcmake/cmake despite missing LLVM source dir" >&2
  exit 1
fi

echo "[wasm-config-behavior] PASS"

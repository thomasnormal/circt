#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${1:-utils/wasm_cxx20_warning_check.sh}"

if [[ ! -x "$SCRIPT" ]]; then
  echo "[wasm-cxx20-warn-behavior] missing executable: $SCRIPT" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

build_dir="$tmpdir/build"
fake_bin="$tmpdir/fake-bin"
mkdir -p "$build_dir" "$fake_bin"

cat >"$build_dir/CMakeCache.txt" <<'EOF'
CMAKE_CXX_STANDARD:STRING=20
EOF

cat >"$fake_bin/ninja" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-C" ]]; then
  shift 2
fi

if [[ "${1:-}" == "-t" && "${2:-}" == "commands" ]]; then
  printf "%s\n" "${NINJA_COMMAND_LINES:-}"
  exit 0
fi

if [[ "${1:-}" == "-t" && "${2:-}" == "clean" ]]; then
  exit 0
fi

exit 0
EOF
chmod +x "$fake_bin/ninja"

run_case() {
  local name="$1"
  local cmd_lines="$2"
  local expect_rc="$3"
  local tmp_out="$tmpdir/$name.out"
  local tmp_err="$tmpdir/$name.err"
  local rc=0

  set +e
  NINJA_COMMAND_LINES="$cmd_lines" \
    PATH="$fake_bin:$PATH" \
    BUILD_DIR="$build_dir" \
    NINJA_JOBS=1 \
    "$SCRIPT" >"$tmp_out" 2>"$tmp_err"
  rc=$?
  set -e

  if [[ "$rc" -ne "$expect_rc" ]]; then
    echo "[wasm-cxx20-warn-behavior] case '$name' failed (rc=$rc expected $expect_rc)" >&2
    cat "$tmp_out" "$tmp_err" >&2
    exit 1
  fi
}

cmd_lines_with_final_cxx20="$(cat <<'EOF'
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/FIRRTLAnnotationsGen.cpp -std=gnu++17 -std=c++20 -o FIRRTLAnnotationsGen.cpp.o
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/FIRRTLIntrinsicsGen.cpp -std=gnu++17 -std=c++20 -o FIRRTLIntrinsicsGen.cpp.o
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/circt-tblgen.cpp -std=gnu++17 -std=c++20 -o circt-tblgen.cpp.o
EOF
)"

cmd_lines_with_final_old_std="$(cat <<'EOF'
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/FIRRTLAnnotationsGen.cpp -std=c++20 -std=gnu++17 -o FIRRTLAnnotationsGen.cpp.o
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/FIRRTLIntrinsicsGen.cpp -std=c++20 -std=gnu++17 -o FIRRTLIntrinsicsGen.cpp.o
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/circt-tblgen.cpp -std=c++20 -std=gnu++17 -o circt-tblgen.cpp.o
EOF
)"

cmd_lines_with_duplicate_src_entries="$(cat <<'EOF'
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/FIRRTLAnnotationsGen.cpp -std=gnu++17 -o FIRRTLAnnotationsGen.cpp.o.old
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/FIRRTLAnnotationsGen.cpp -std=c++20 -o FIRRTLAnnotationsGen.cpp.o
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/FIRRTLIntrinsicsGen.cpp -std=gnu++17 -o FIRRTLIntrinsicsGen.cpp.o.old
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/FIRRTLIntrinsicsGen.cpp -std=c++20 -o FIRRTLIntrinsicsGen.cpp.o
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/circt-tblgen.cpp -std=gnu++17 -o circt-tblgen.cpp.o.old
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/circt-tblgen.cpp -std=c++20 -o circt-tblgen.cpp.o
EOF
)"

cmd_lines_with_trailing_link_step="$(cat <<'EOF'
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/FIRRTLAnnotationsGen.cpp -std=c++20 -o FIRRTLAnnotationsGen.cpp.o
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/FIRRTLIntrinsicsGen.cpp -std=c++20 -o FIRRTLIntrinsicsGen.cpp.o
/opt/emsdk/upstream/emscripten/em++ -c tools/circt-tblgen/circt-tblgen.cpp -std=c++20 -o circt-tblgen.cpp.o
/opt/emsdk/upstream/emscripten/em++ tools/circt-tblgen/FIRRTLAnnotationsGen.cpp.o tools/circt-tblgen/FIRRTLIntrinsicsGen.cpp.o tools/circt-tblgen/circt-tblgen.cpp.o -o bin/circt-tblgen.js
EOF
)"

# Effective standard should be taken from the last -std flag.
run_case "final-cxx20" "$cmd_lines_with_final_cxx20" 0
run_case "final-old-std" "$cmd_lines_with_final_old_std" 1
run_case "duplicate-src-last-command-wins" "$cmd_lines_with_duplicate_src_entries" 0
run_case "trailing-link-step-ignored" "$cmd_lines_with_trailing_link_step" 0

echo "[wasm-cxx20-warn-behavior] PASS"

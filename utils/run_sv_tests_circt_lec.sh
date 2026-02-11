#!/usr/bin/env bash
set -euo pipefail

SV_TESTS_DIR="${1:-/home/thomas-ahle/sv-tests}"
TAG_REGEX="${TAG_REGEX:-}"

# Memory limit settings to prevent system hangs
CIRCT_MEMORY_LIMIT_GB="${CIRCT_MEMORY_LIMIT_GB:-20}"
CIRCT_TIMEOUT_SECS="${CIRCT_TIMEOUT_SECS:-300}"
CIRCT_MEMORY_LIMIT_KB=$((CIRCT_MEMORY_LIMIT_GB * 1024 * 1024))

# Run a command with memory limit
run_limited() {
  (
    ulimit -v $CIRCT_MEMORY_LIMIT_KB 2>/dev/null || true
    timeout --signal=KILL $CIRCT_TIMEOUT_SECS "$@"
  )
}
TEST_FILTER="${TEST_FILTER:-}"
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
LEC_MLIR_CACHE_DIR="${LEC_MLIR_CACHE_DIR:-}"
CIRCT_OPT="${CIRCT_OPT:-build/bin/circt-opt}"
CIRCT_LEC="${CIRCT_LEC:-build/bin/circt-lec}"
CIRCT_OPT_ARGS="${CIRCT_OPT_ARGS:-}"
CIRCT_LEC_ARGS="${CIRCT_LEC_ARGS:-}"
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UVM_TAG_REGEX="${UVM_TAG_REGEX:-(^| )uvm( |$)}"
INCLUDE_UVM_TAGS="${INCLUDE_UVM_TAGS:-0}"
TAG_REGEX_EFFECTIVE="$TAG_REGEX"
if [[ "$INCLUDE_UVM_TAGS" == "1" ]]; then
  TAG_REGEX_EFFECTIVE="($TAG_REGEX_EFFECTIVE)|$UVM_TAG_REGEX"
fi
LEC_SMOKE_ONLY="${LEC_SMOKE_ONLY:-0}"
# LEC_FAIL_ON_INEQ removed - circt-lec no longer has --fail-on-inequivalent flag
FORCE_LEC="${FORCE_LEC:-0}"
SKIP_SHOULD_FAIL="${SKIP_SHOULD_FAIL:-1}"
Z3_BIN="${Z3_BIN:-}"
OUT="${OUT:-$PWD/sv-tests-lec-results.txt}"
mkdir -p "$(dirname "$OUT")" 2>/dev/null || true
KEEP_LOGS_DIR="${KEEP_LOGS_DIR:-}"
LEC_ASSUME_KNOWN_INPUTS="${LEC_ASSUME_KNOWN_INPUTS:-0}"
LEC_ACCEPT_XPROP_ONLY="${LEC_ACCEPT_XPROP_ONLY:-0}"
DROP_REMARK_PATTERN="${DROP_REMARK_PATTERN:-will be dropped during lowering}"
LEC_DROP_REMARK_CASES_OUT="${LEC_DROP_REMARK_CASES_OUT:-}"
LEC_DROP_REMARK_REASONS_OUT="${LEC_DROP_REMARK_REASONS_OUT:-}"

resolve_default_uvm_path() {
  local candidate
  for candidate in \
    "$SCRIPT_DIR/../lib/Runtime/uvm" \
    "$SCRIPT_DIR/../lib/Runtime/uvm-core/src" \
    "$SCRIPT_DIR/../lib/Runtime/uvm-core" \
    "/home/thomas-ahle/uvm-core/src"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "$SCRIPT_DIR/../lib/Runtime/uvm"
}

UVM_PATH="${UVM_PATH:-$(resolve_default_uvm_path)}"

if [[ ! -d "$SV_TESTS_DIR/tests" ]]; then
  echo "sv-tests directory not found: $SV_TESTS_DIR" >&2
  exit 1
fi

if [[ -z "$TAG_REGEX" && -z "$TEST_FILTER" ]]; then
  echo "must set TAG_REGEX or TEST_FILTER explicitly (no default filter)" >&2
  exit 1
fi

if [[ "$LEC_SMOKE_ONLY" != "1" ]]; then
  if [[ -n "$Z3_BIN" ]]; then
    if [[ "$Z3_BIN" == */* ]]; then
      if [[ ! -x "$Z3_BIN" ]]; then
        echo "z3 not found or not executable: $Z3_BIN" >&2
        exit 1
      fi
    elif ! command -v "$Z3_BIN" >/dev/null 2>&1; then
      echo "z3 not found in PATH: $Z3_BIN" >&2
      exit 1
    fi
  else
    if command -v z3 >/dev/null 2>&1; then
      Z3_BIN="z3"
    elif [[ -x /home/thomas-ahle/z3-install/bin/z3 ]]; then
      Z3_BIN="/home/thomas-ahle/z3-install/bin/z3"
    elif [[ -x /home/thomas-ahle/z3/build/z3 ]]; then
      Z3_BIN="/home/thomas-ahle/z3/build/z3"
    fi
  fi

  if [[ -z "$Z3_BIN" ]]; then
    echo "z3 not found; set Z3_BIN or enable LEC_SMOKE_ONLY" >&2
    exit 1
  fi
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

results_tmp="$tmpdir/results.txt"
touch "$results_tmp"

pass=0
fail=0
error=0
skip=0
total=0
drop_remark_cases=0
cache_hits=0
cache_misses=0
cache_stores=0

declare -A drop_remark_seen_cases
declare -A drop_remark_seen_case_reasons

record_drop_remark_case() {
  local case_id="$1"
  local case_path="$2"
  local verilog_log="$3"
  if [[ -z "$case_id" || ! -s "$verilog_log" ]]; then
    return
  fi
  if ! grep -Fq "$DROP_REMARK_PATTERN" "$verilog_log"; then
    return
  fi
  while IFS= read -r reason; do
    if [[ -z "$reason" ]]; then
      continue
    fi
    local reason_key="${case_id}|${reason}"
    if [[ -n "${drop_remark_seen_case_reasons["$reason_key"]+x}" ]]; then
      continue
    fi
    drop_remark_seen_case_reasons["$reason_key"]=1
    if [[ -n "$LEC_DROP_REMARK_REASONS_OUT" ]]; then
      printf "%s\t%s\t%s\n" "$case_id" "$case_path" "$reason" >> "$LEC_DROP_REMARK_REASONS_OUT"
    fi
  done < <(
    awk -v pattern="$DROP_REMARK_PATTERN" '
      index($0, pattern) {
        line = $0
        gsub(/\t/, " ", line)
        sub(/^[[:space:]]+/, "", line)
        if (match(line, /^[^:]+:[0-9]+(:[0-9]+)?:[[:space:]]*/))
          line = substr(line, RLENGTH + 1)
        sub(/^[Ww]arning:[[:space:]]*/, "", line)
        gsub(/[[:space:]]+/, " ", line)
        gsub(/[0-9]+/, "<n>", line)
        gsub(/;/, ",", line)
        if (length(line) > 240)
          line = substr(line, 1, 240)
        print line
      }
    ' "$verilog_log" | sort -u
  )
  if [[ -n "${drop_remark_seen_cases["$case_id"]+x}" ]]; then
    return
  fi
  drop_remark_seen_cases["$case_id"]=1
  drop_remark_cases=$((drop_remark_cases + 1))
  if [[ -n "$LEC_DROP_REMARK_CASES_OUT" ]]; then
    printf "%s\t%s\n" "$case_id" "$case_path" >> "$LEC_DROP_REMARK_CASES_OUT"
  fi
}

read_meta() {
  local key="$1"
  local file="$2"
  sed -n "s/^[[:space:]]*:${key}:[[:space:]]*//p" "$file" | head -n 1
}

normalize_paths() {
  local root="$1"
  shift
  local out=()
  for item in "$@"; do
    if [[ -z "$item" ]]; then
      continue
    fi
    if [[ "$item" = /* ]]; then
      out+=("$item")
    else
      out+=("$root/$item")
    fi
  done
  printf '%s\n' "${out[@]}"
}

hash_key() {
  local payload="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    printf "%s" "$payload" | sha256sum | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    printf "%s" "$payload" | shasum -a 256 | awk '{print $1}'
  else
    printf "%s" "$payload" | cksum | awk '{print $1}'
  fi
}

hash_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    cksum "$path" | awk '{print $1}'
  fi
}

save_logs() {
  if [[ -z "$KEEP_LOGS_DIR" ]]; then
    return
  fi
  mkdir -p "$KEEP_LOGS_DIR"
  cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
  cp -f "$opt_mlir" "$KEEP_LOGS_DIR/${log_tag}.opt.mlir" 2>/dev/null || true
  cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
    2>/dev/null || true
  cp -f "$opt_log" "$KEEP_LOGS_DIR/${log_tag}.circt-opt.log" 2>/dev/null || true
  cp -f "$lec_log" "$KEEP_LOGS_DIR/${log_tag}.circt-lec.log" 2>/dev/null || true
}

extract_lec_diag() {
  local lec_text="$1"
  if [[ "$lec_text" =~ LEC_DIAG=([A-Z0-9_]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  fi
}

extract_lec_result_tag() {
  local lec_text="$1"
  if [[ "$lec_text" =~ LEC_RESULT=([A-Z0-9_]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  if grep -Fq "c1 == c2" <<<"$lec_text"; then
    printf 'EQ\n'
    return 0
  fi
  if grep -Fq "c1 != c2" <<<"$lec_text"; then
    printf 'NEQ\n'
    return 0
  fi
}

while IFS= read -r -d '' sv; do
  tags="$(read_meta tags "$sv")"
  if [[ -z "$tags" ]]; then
    skip=$((skip + 1))
    continue
  fi
  if ! [[ "$tags" =~ $TAG_REGEX_EFFECTIVE ]]; then
    skip=$((skip + 1))
    continue
  fi

  base="$(basename "$sv" .sv)"
  if [[ -n "$TEST_FILTER" ]] && ! [[ "$base" =~ $TEST_FILTER ]]; then
    skip=$((skip + 1))
    continue
  fi

  type="$(read_meta type "$sv")"
  run_lec=1
  if [[ "$type" =~ [Pp]arsing ]]; then
    run_lec=0
    if [[ "$FORCE_LEC" == "1" ]]; then
      run_lec=1
    fi
  fi

  should_fail="$(read_meta should_fail "$sv")"
  should_fail_because="$(read_meta should_fail_because "$sv")"
  if [[ -n "$should_fail_because" ]]; then
    should_fail="1"
  fi
  if [[ "$SKIP_SHOULD_FAIL" == "1" && "$should_fail" == "1" ]]; then
    skip=$((skip + 1))
    continue
  fi

  files_line="$(read_meta files "$sv")"
  incdirs_line="$(read_meta incdirs "$sv")"
  defines_line="$(read_meta defines "$sv")"
  top_module="$(read_meta top_module "$sv")"
  if [[ -z "$top_module" ]]; then
    top_module="top"
  fi

  test_root="$SV_TESTS_DIR/tests"
  log_tag="$base"
  rel_path="${sv#"$test_root/"}"
  if [[ "$rel_path" != "$sv" ]]; then
    log_tag="${rel_path%.sv}"
  fi
  log_tag="${log_tag//\//__}"
  if [[ -z "$files_line" ]]; then
    files=("$sv")
  else
    mapfile -t files < <(normalize_paths "$test_root" $files_line)
  fi
  if [[ -z "$incdirs_line" ]]; then
    incdirs=()
  else
    mapfile -t incdirs < <(normalize_paths "$test_root" $incdirs_line)
  fi
  incdirs+=("$(dirname "$sv")")

  defines=()
  if [[ -n "$defines_line" ]]; then
    for d in $defines_line; do
      defines+=("$d")
    done
  fi

  use_uvm=0
  if [[ "$tags" =~ $UVM_TAG_REGEX ]]; then
    use_uvm=1
  fi
  if [[ "$use_uvm" == "1" && ! -d "$UVM_PATH" ]]; then
    printf "ERROR\t%s\t%s (UVM path not found: %s)\tsv-tests\tLEC\tUVM_PATH_MISSING\n" \
      "$base" "$sv" "$UVM_PATH" >> "$results_tmp"
    error=$((error + 1))
    total=$((total + 1))
    continue
  fi

  total=$((total + 1))

  mlir="$tmpdir/${base}.mlir"
  opt_mlir="$tmpdir/${base}.opt.mlir"
  verilog_log="$tmpdir/${base}.circt-verilog.log"
  opt_log="$tmpdir/${base}.circt-opt.log"
  lec_log="$tmpdir/${base}.circt-lec.log"

  cmd=("$CIRCT_VERILOG" --ir-hw --timescale=1ns/1ns --single-unit \
    -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob)
  if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" && "$use_uvm" == "0" ]]; then
    cmd+=("--no-uvm-auto-include")
  fi
  if [[ "$use_uvm" == "1" ]]; then
    cmd+=("--uvm-path=$UVM_PATH")
  fi
  if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
    read -r -a extra_verilog_args <<<"$CIRCT_VERILOG_ARGS"
    cmd+=("${extra_verilog_args[@]}")
  fi
  for inc in "${incdirs[@]}"; do
    cmd+=("-I" "$inc")
  done
  for def in "${defines[@]}"; do
    cmd+=("-D" "$def")
  done
  if [[ -n "$top_module" ]]; then
    cmd+=("--top=$top_module")
  fi
  cmd+=("${files[@]}")

  cache_hit=0
  cache_file=""
  if [[ -n "$LEC_MLIR_CACHE_DIR" ]]; then
    mkdir -p "$LEC_MLIR_CACHE_DIR"
    cache_payload="circt_verilog=${CIRCT_VERILOG}
circt_verilog_args=${CIRCT_VERILOG_ARGS}
disable_uvm_auto_include=${DISABLE_UVM_AUTO_INCLUDE}
use_uvm=${use_uvm}
uvm_path=${UVM_PATH}
top_module=${top_module}
"
    for inc in "${incdirs[@]}"; do
      cache_payload+="incdir=${inc}
"
    done
    for def in "${defines[@]}"; do
      cache_payload+="define=${def}
"
    done
    for f in "${files[@]}"; do
      cache_payload+="file=${f} hash=$(hash_file "$f")
"
    done
    cache_key="$(hash_key "$cache_payload")"
    cache_file="$LEC_MLIR_CACHE_DIR/${cache_key}.mlir"
    if [[ -s "$cache_file" ]]; then
      cp -f "$cache_file" "$mlir"
      : > "$verilog_log"
      echo "[run_sv_tests_circt_lec] cache-hit key=$cache_key case=$base" >> "$verilog_log"
      cache_hit=1
      cache_hits=$((cache_hits + 1))
    else
      cache_misses=$((cache_misses + 1))
    fi
  fi

  if [[ "$cache_hit" != "1" ]]; then
    if run_limited "${cmd[@]}" > "$mlir" 2> "$verilog_log"; then
      :
    else
      verilog_status=$?
      record_drop_remark_case "$base" "$sv" "$verilog_log"
      if [[ "$verilog_status" -eq 124 || "$verilog_status" -eq 137 ]]; then
        printf "TIMEOUT\t%s\t%s\tsv-tests\tLEC\tCIRCT_VERILOG_TIMEOUT\tpreprocess\n" "$base" "$sv" >> "$results_tmp"
      else
        printf "ERROR\t%s\t%s\tsv-tests\tLEC\tCIRCT_VERILOG_ERROR\n" "$base" "$sv" >> "$results_tmp"
      fi
      error=$((error + 1))
      save_logs
      continue
    fi
    record_drop_remark_case "$base" "$sv" "$verilog_log"
    if [[ -n "$cache_file" && -s "$mlir" ]]; then
      cache_tmp="$cache_file.tmp.$$.$RANDOM"
      cp -f "$mlir" "$cache_tmp" 2>/dev/null || true
      mv -f "$cache_tmp" "$cache_file" 2>/dev/null || true
      cache_stores=$((cache_stores + 1))
    fi
  fi

  opt_cmd=("$CIRCT_OPT" --lower-llhd-ref-ports --strip-llhd-processes
    --strip-llhd-interface-signals --lower-ltl-to-core
    --lower-clocked-assert-like)
  if [[ -n "$CIRCT_OPT_ARGS" ]]; then
    read -r -a extra_opt_args <<<"$CIRCT_OPT_ARGS"
    opt_cmd+=("${extra_opt_args[@]}")
  fi
  opt_cmd+=("$mlir")

  if run_limited "${opt_cmd[@]}" > "$opt_mlir" 2> "$opt_log"; then
    :
  else
    opt_status=$?
    if [[ ! -s "$opt_log" ]]; then
      printf "error: circt-opt failed without diagnostics for case '%s'\n" \
        "$base" | tee -a "$opt_log" >&2
    fi
    if [[ "$opt_status" -eq 124 || "$opt_status" -eq 137 ]]; then
      printf "TIMEOUT\t%s\t%s\tsv-tests\tLEC\tCIRCT_OPT_TIMEOUT\tpreprocess\n" "$base" "$sv" >> "$results_tmp"
    else
      printf "ERROR\t%s\t%s\tsv-tests\tLEC\tCIRCT_OPT_ERROR\n" "$base" "$sv" >> "$results_tmp"
    fi
    error=$((error + 1))
    save_logs
    continue
  fi

  if [[ "$run_lec" == "0" ]]; then
    printf "PASS\t%s\t%s\tsv-tests\tLEC\tLEC_NOT_RUN\n" "$base" "$sv" >> "$results_tmp"
    pass=$((pass + 1))
    save_logs
    continue
  fi

  lec_args=()
  if [[ "$LEC_SMOKE_ONLY" == "1" ]]; then
    lec_args+=("--emit-mlir")
  else
    lec_args+=("--run-smtlib" "--z3-path=$Z3_BIN")
    # Note: --fail-on-inequivalent was removed from circt-lec
    # Result checking is done via output parsing below
  fi
  if [[ "$LEC_ASSUME_KNOWN_INPUTS" == "1" ]]; then
    lec_args+=("--assume-known-inputs")
  fi
  if [[ "$LEC_ACCEPT_XPROP_ONLY" == "1" ]]; then
    lec_args+=("--accept-xprop-only")
  fi
  if [[ -n "$CIRCT_LEC_ARGS" ]]; then
    read -r -a extra_lec_args <<<"$CIRCT_LEC_ARGS"
    lec_args+=("${extra_lec_args[@]}")
  fi
  lec_args+=("-c1=$top_module" "-c2=$top_module" "$opt_mlir" "$opt_mlir")

  lec_out=""
  if lec_out="$(run_limited "$CIRCT_LEC" "${lec_args[@]}" 2> "$lec_log")"; then
    lec_status=0
  else
    lec_status=$?
  fi
  lec_combined="$lec_out"
  if [[ -s "$lec_log" ]]; then
    lec_combined+=$'\n'"$(cat "$lec_log")"
  fi
  lec_diag="$(extract_lec_diag "$lec_combined")"

  if [[ "$LEC_SMOKE_ONLY" == "1" ]]; then
    if [[ "$lec_status" -eq 0 ]]; then
      result="PASS"
    else
      result="ERROR"
    fi
  else
    if grep -q "LEC_RESULT=EQ" <<<"$lec_combined"; then
      result="PASS"
    elif grep -q "LEC_RESULT=NEQ" <<<"$lec_combined"; then
      result="FAIL"
    elif grep -Fq "c1 == c2" <<<"$lec_combined"; then
      result="PASS"
    elif grep -Fq "c1 != c2" <<<"$lec_combined"; then
      result="FAIL"
    else
      result="ERROR"
    fi
  fi

  if [[ -z "$lec_diag" ]]; then
    lec_diag="$(extract_lec_result_tag "$lec_combined")"
  fi
  if [[ -z "$lec_diag" ]]; then
    case "$result" in
      PASS) lec_diag="EQ" ;;
      FAIL) lec_diag="NEQ" ;;
      *)
        if [[ "$lec_status" -eq 124 || "$lec_status" -eq 137 ]]; then
          lec_diag="TIMEOUT"
          result="TIMEOUT"
        else
          lec_diag="ERROR"
        fi
        ;;
    esac
  fi

  case "$result" in
    PASS) pass=$((pass + 1)) ;;
    FAIL) fail=$((fail + 1)) ;;
    *) error=$((error + 1)) ;;
  esac
  lec_timeout_class=""
  if [[ "$result" == "TIMEOUT" ]]; then
    lec_timeout_class="solver_budget"
  fi
  if [[ -n "$lec_timeout_class" ]]; then
    printf "%s\t%s\t%s\tsv-tests\tLEC\t%s\t%s\n" "$result" "$base" "$sv" "$lec_diag" "$lec_timeout_class" >> "$results_tmp"
  else
    printf "%s\t%s\t%s\tsv-tests\tLEC\t%s\n" "$result" "$base" "$sv" "$lec_diag" >> "$results_tmp"
  fi
  save_logs
done < <(find "$SV_TESTS_DIR/tests" -type f -name "*.sv" -print0)

sort "$results_tmp" > "$OUT"

if [[ -n "$LEC_DROP_REMARK_CASES_OUT" && -f "$LEC_DROP_REMARK_CASES_OUT" ]]; then
  sort -u -o "$LEC_DROP_REMARK_CASES_OUT" "$LEC_DROP_REMARK_CASES_OUT"
fi
if [[ -n "$LEC_DROP_REMARK_REASONS_OUT" && -f "$LEC_DROP_REMARK_REASONS_OUT" ]]; then
  sort -u -o "$LEC_DROP_REMARK_REASONS_OUT" "$LEC_DROP_REMARK_REASONS_OUT"
fi

echo "sv-tests LEC summary: total=$total pass=$pass fail=$fail error=$error skip=$skip"
echo "sv-tests LEC dropped-syntax summary: drop_remark_cases=$drop_remark_cases pattern='$DROP_REMARK_PATTERN'"
echo "sv-tests frontend cache summary: hits=$cache_hits misses=$cache_misses stores=$cache_stores"
echo "results: $OUT"

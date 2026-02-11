#!/usr/bin/env bash
set -euo pipefail

VERIF_DIR="${1:-/home/thomas-ahle/verilator-verification}"
shift || true
TOP="${TOP:-top}"
TEST_FILTER="${TEST_FILTER:-}"
CIRCT_TIMEOUT_SECS="${CIRCT_TIMEOUT_SECS:-300}"
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
LEC_MLIR_CACHE_DIR="${LEC_MLIR_CACHE_DIR:-}"
CIRCT_OPT="${CIRCT_OPT:-build/bin/circt-opt}"
CIRCT_LEC="${CIRCT_LEC:-build/bin/circt-lec}"
CIRCT_OPT_ARGS="${CIRCT_OPT_ARGS:-}"
CIRCT_LEC_ARGS="${CIRCT_LEC_ARGS:-}"
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
LEC_SMOKE_ONLY="${LEC_SMOKE_ONLY:-0}"
LEC_ACCEPT_XPROP_ONLY="${LEC_ACCEPT_XPROP_ONLY:-0}"
Z3_BIN="${Z3_BIN:-}"
OUT="${OUT:-$PWD/verilator-verification-lec-results.txt}"
mkdir -p "$(dirname "$OUT")" 2>/dev/null || true
KEEP_LOGS_DIR="${KEEP_LOGS_DIR:-}"
DROP_REMARK_PATTERN="${DROP_REMARK_PATTERN:-will be dropped during lowering}"
LEC_DROP_REMARK_CASES_OUT="${LEC_DROP_REMARK_CASES_OUT:-}"
LEC_DROP_REMARK_REASONS_OUT="${LEC_DROP_REMARK_REASONS_OUT:-}"

run_limited() {
  timeout --signal=KILL "$CIRCT_TIMEOUT_SECS" "$@"
}

if [[ ! -d "$VERIF_DIR/tests" ]]; then
  echo "verilator-verification directory not found: $VERIF_DIR" >&2
  exit 1
fi

if [[ -z "$TEST_FILTER" ]]; then
  echo "must set TEST_FILTER explicitly (no default filter)" >&2
  exit 1
fi

if [[ -z "$Z3_BIN" ]]; then
  if command -v z3 >/dev/null 2>&1; then
    Z3_BIN="z3"
  elif [[ -x /home/thomas-ahle/z3-install/bin/z3 ]]; then
    Z3_BIN="/home/thomas-ahle/z3-install/bin/z3"
  elif [[ -x /home/thomas-ahle/z3/build/z3 ]]; then
    Z3_BIN="/home/thomas-ahle/z3/build/z3"
  else
    Z3_BIN="z3"
  fi
fi

suites=("$@")
if [[ ${#suites[@]} -eq 0 ]]; then
  suites=("$VERIF_DIR/tests/asserts" "$VERIF_DIR/tests/sequences" \
    "$VERIF_DIR/tests/event-control-expression")
else
  for i in "${!suites[@]}"; do
    if [[ "${suites[$i]}" != /* ]]; then
      suites[$i]="$VERIF_DIR/${suites[$i]}"
    fi
  done
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

# Detect top module similar to the BMC harness.
detect_top() {
  local file="$1"
  local requested="$2"
  local modules=()
  while IFS= read -r name; do
    modules+=("$name")
  done < <(awk '
    /^[[:space:]]*module[[:space:]]+/ {
      name=$2
      sub(/[^A-Za-z0-9_$].*/, "", name)
      if (name != "") print name
    }' "$file")

  if [[ -n "$requested" ]]; then
    for m in "${modules[@]}"; do
      if [[ "$m" == "$requested" ]]; then
        echo "$requested"
        return
      fi
    done
  fi

  if [[ ${#modules[@]} -eq 1 ]]; then
    echo "${modules[0]}"
    return
  fi

  if [[ -z "$requested" || "$requested" == "top" ]]; then
    if [[ ${#modules[@]} -gt 0 ]]; then
      echo "${modules[0]}"
      return
    fi
  fi

  echo "$requested"
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

extract_opt_error_reason() {
  local opt_log_file="$1"
  if [[ ! -s "$opt_log_file" ]]; then
    printf 'no_diag\n'
    return 0
  fi
  awk '
    NF {
      line = $0
      gsub(/\t/, " ", line)
      sub(/^[[:space:]]+/, "", line)
      if (match(line, /^[^:]+:[0-9]+(:[0-9]+)?:[[:space:]]*/))
        line = substr(line, RLENGTH + 1)
      sub(/^[Ee]rror:[[:space:]]*/, "", line)
      low = tolower(line)
      if (index(low, "circt-opt failed without diagnostics for case") > 0) {
        print "no_diag"
        exit
      }
      gsub(/[0-9]+/, "<n>", line)
      gsub(/[^A-Za-z0-9]+/, "_", line)
      gsub(/^_+/, "", line)
      gsub(/_+$/, "", line)
      line = tolower(line)
      if (length(line) == 0)
        line = "no_diag"
      if (length(line) > 64)
        line = substr(line, 1, 64)
      print line
      exit
    }
    END {
      if (NR == 0)
        print "no_diag"
    }
  ' "$opt_log_file"
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

for suite in "${suites[@]}"; do
  if [[ ! -d "$suite" ]]; then
    echo "suite not found: $suite" >&2
    continue
  fi
  while IFS= read -r -d '' sv; do
    base="$(basename "$sv" .sv)"
    if [[ -n "$TEST_FILTER" ]] && ! [[ "$base" =~ $TEST_FILTER ]]; then
      skip=$((skip + 1))
      continue
    fi

    log_tag="$base"
    rel_path="${sv#"$VERIF_DIR/"}"
    if [[ "$rel_path" != "$sv" ]]; then
      log_tag="${rel_path%.sv}"
    fi
    log_tag="${log_tag//\//__}"

    total=$((total + 1))

    mlir="$tmpdir/${base}.mlir"
    opt_mlir="$tmpdir/${base}.opt.mlir"
    verilog_log="$tmpdir/${base}.circt-verilog.log"
    opt_log="$tmpdir/${base}.circt-opt.log"
    lec_log="$tmpdir/${base}.circt-lec.log"
    top_for_file="$(detect_top "$sv" "$TOP")"

    cmd=("$CIRCT_VERILOG" --ir-hw --timescale=1ns/1ns --single-unit \
      -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob)
    if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" ]]; then
      cmd+=("--no-uvm-auto-include")
    fi
    if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
      read -r -a extra_verilog_args <<<"$CIRCT_VERILOG_ARGS"
      cmd+=("${extra_verilog_args[@]}")
    fi
    cmd+=("-I" "$(dirname "$sv")")
    if [[ -n "$top_for_file" ]]; then
      cmd+=("--top=$top_for_file")
    fi
    cmd+=("$sv")

    cache_hit=0
    cache_file=""
    if [[ -n "$LEC_MLIR_CACHE_DIR" ]]; then
      mkdir -p "$LEC_MLIR_CACHE_DIR"
      cache_payload="runner=verilator_lec
circt_verilog=${CIRCT_VERILOG}
circt_verilog_args=${CIRCT_VERILOG_ARGS}
disable_uvm_auto_include=${DISABLE_UVM_AUTO_INCLUDE}
top_module=${top_for_file}
incdir=$(dirname "$sv")
file=${sv} hash=$(hash_file "$sv")
"
      cache_key="$(hash_key "$cache_payload")"
      cache_file="$LEC_MLIR_CACHE_DIR/${cache_key}.mlir"
      if [[ -s "$cache_file" ]]; then
        cp -f "$cache_file" "$mlir"
        : > "$verilog_log"
        echo "[run_verilator_verification_circt_lec] cache-hit key=$cache_key case=$base" >> "$verilog_log"
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
          printf "TIMEOUT\t%s\t%s\tverilator-verification\tLEC\tCIRCT_VERILOG_TIMEOUT\tpreprocess\n" "$base" "$sv" >> "$results_tmp"
        else
          printf "ERROR\t%s\t%s\tverilator-verification\tLEC\tCIRCT_VERILOG_ERROR\n" "$base" "$sv" >> "$results_tmp"
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
        printf "TIMEOUT\t%s\t%s\tverilator-verification\tLEC\tCIRCT_OPT_TIMEOUT\tpreprocess\n" "$base" "$sv" >> "$results_tmp"
      else
        opt_reason="$(extract_opt_error_reason "$opt_log")"
        printf "ERROR\t%s\t%s\tverilator-verification\tLEC\tCIRCT_OPT_ERROR\t%s\n" "$base" "$sv" "$opt_reason" >> "$results_tmp"
      fi
      error=$((error + 1))
      save_logs
      continue
    fi

    lec_args=()
    if [[ "$LEC_SMOKE_ONLY" == "1" ]]; then
      lec_args+=("--emit-mlir")
    else
      lec_args+=("--run-smtlib" "--z3-path=$Z3_BIN")
    fi
    if [[ "$LEC_ACCEPT_XPROP_ONLY" == "1" ]]; then
      lec_args+=("--accept-xprop-only")
    fi
    if [[ -n "$CIRCT_LEC_ARGS" ]]; then
      read -r -a extra_lec_args <<<"$CIRCT_LEC_ARGS"
      lec_args+=("${extra_lec_args[@]}")
    fi
    lec_args+=("-c1=$top_for_file" "-c2=$top_for_file" "$opt_mlir" "$opt_mlir")

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
      elif grep -q "c1 == c2" <<<"$lec_combined"; then
        result="PASS"
      elif grep -q "c1 != c2" <<<"$lec_combined"; then
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
      printf "%s\t%s\t%s\tverilator-verification\tLEC\t%s\t%s\n" "$result" "$base" "$sv" "$lec_diag" "$lec_timeout_class" >> "$results_tmp"
    else
      printf "%s\t%s\t%s\tverilator-verification\tLEC\t%s\n" "$result" "$base" "$sv" "$lec_diag" >> "$results_tmp"
    fi
    save_logs
  done < <(find "$suite" -type f -name "*.sv" -print0)
done

sort "$results_tmp" > "$OUT"

if [[ -n "$LEC_DROP_REMARK_CASES_OUT" && -f "$LEC_DROP_REMARK_CASES_OUT" ]]; then
  sort -u -o "$LEC_DROP_REMARK_CASES_OUT" "$LEC_DROP_REMARK_CASES_OUT"
fi
if [[ -n "$LEC_DROP_REMARK_REASONS_OUT" && -f "$LEC_DROP_REMARK_REASONS_OUT" ]]; then
  sort -u -o "$LEC_DROP_REMARK_REASONS_OUT" "$LEC_DROP_REMARK_REASONS_OUT"
fi

echo "verilator-verification LEC summary: total=$total pass=$pass fail=$fail error=$error skip=$skip"
echo "verilator-verification LEC dropped-syntax summary: drop_remark_cases=$drop_remark_cases pattern='$DROP_REMARK_PATTERN'"
echo "verilator-verification frontend cache summary: hits=$cache_hits misses=$cache_misses stores=$cache_stores"
echo "results: $OUT"

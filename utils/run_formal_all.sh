#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
run_formal_all.sh [options]

Runs formal BMC/LEC suites and summarizes results.

Options:
  --out-dir DIR          Output directory for logs/results (default: formal-results-YYYYMMDD)
  --sv-tests DIR         sv-tests root (default: ~/sv-tests)
  --verilator DIR        verilator-verification root (default: ~/verilator-verification)
  --yosys DIR            yosys/tests/sva root (default: ~/yosys/tests/sva)
  --z3-bin PATH          Path to z3 binary (optional)
  --baseline-file FILE   Baseline TSV file (default: utils/formal-baselines.tsv)
  --plan-file FILE       Project plan file to update (default: PROJECT_PLAN.md)
  --update-baselines     Update baseline file and PROJECT_PLAN.md table
  --fail-on-diff         Fail if results differ from baseline file
  --strict-gate          Fail on new fail/error/xpass and pass-rate regression vs baseline
  --baseline-window N    Baseline rows per suite/mode used for gate comparison
                         (default: 1, latest baseline only)
  --fail-on-new-xpass    Fail when xpass count increases vs baseline
  --fail-on-passrate-regression
                         Fail when pass-rate decreases vs baseline
  --json-summary FILE    Write machine-readable JSON summary (default: <out-dir>/summary.json)
  --bmc-run-smtlib        Use circt-bmc --run-smtlib (external z3) in suite runs
  --bmc-assume-known-inputs  Add --assume-known-inputs to BMC runs
  --lec-assume-known-inputs  Add --assume-known-inputs to LEC runs
  --lec-accept-xprop-only    Treat XPROP_ONLY mismatches as equivalent in LEC runs
  --with-opentitan       Run OpenTitan LEC script
  --opentitan DIR        OpenTitan root (default: ~/opentitan)
  --circt-verilog PATH   Path to circt-verilog (default: <repo>/build/bin/circt-verilog)
  --circt-verilog-avip PATH
                         Path override for AVIP runs (default: --circt-verilog value)
  --circt-verilog-opentitan PATH
                         Path override for OpenTitan runs (default: --circt-verilog value)
  --with-avip            Run AVIP compile smoke using run_avip_circt_verilog.sh
  --avip-glob GLOB       Glob for AVIP dirs (default: ~/mbit/*avip*)
  -h, --help             Show this help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATE_STR="$(date +%Y-%m-%d)"
OUT_DIR=""
SV_TESTS_DIR="${HOME}/sv-tests"
VERILATOR_DIR="${HOME}/verilator-verification"
YOSYS_DIR="${HOME}/yosys/tests/sva"
OPENTITAN_DIR="${HOME}/opentitan"
AVIP_GLOB="${HOME}/mbit/*avip*"
CIRCT_VERILOG_BIN="$REPO_ROOT/build/bin/circt-verilog"
CIRCT_VERILOG_BIN_AVIP=""
CIRCT_VERILOG_BIN_OPENTITAN=""
BASELINE_FILE="utils/formal-baselines.tsv"
PLAN_FILE="PROJECT_PLAN.md"
Z3_BIN="${Z3_BIN:-}"
UPDATE_BASELINES=0
FAIL_ON_DIFF=0
STRICT_GATE=0
BASELINE_WINDOW=1
FAIL_ON_NEW_XPASS=0
FAIL_ON_PASSRATE_REGRESSION=0
JSON_SUMMARY_FILE=""
WITH_OPENTITAN=0
WITH_AVIP=0
BMC_RUN_SMTLIB=0
BMC_ASSUME_KNOWN_INPUTS=0
LEC_ASSUME_KNOWN_INPUTS=0
LEC_ACCEPT_XPROP_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --sv-tests)
      SV_TESTS_DIR="$2"; shift 2 ;;
    --verilator)
      VERILATOR_DIR="$2"; shift 2 ;;
    --yosys)
      YOSYS_DIR="$2"; shift 2 ;;
    --z3-bin)
      Z3_BIN="$2"; shift 2 ;;
    --baseline-file)
      BASELINE_FILE="$2"; shift 2 ;;
    --plan-file)
      PLAN_FILE="$2"; shift 2 ;;
    --opentitan)
      OPENTITAN_DIR="$2"; WITH_OPENTITAN=1; shift 2 ;;
    --with-opentitan)
      WITH_OPENTITAN=1; shift ;;
    --circt-verilog)
      CIRCT_VERILOG_BIN="$2"; shift 2 ;;
    --circt-verilog-avip)
      CIRCT_VERILOG_BIN_AVIP="$2"; shift 2 ;;
    --circt-verilog-opentitan)
      CIRCT_VERILOG_BIN_OPENTITAN="$2"; shift 2 ;;
    --with-avip)
      WITH_AVIP=1; shift ;;
    --avip-glob)
      AVIP_GLOB="$2"; shift 2 ;;
    --bmc-run-smtlib)
      BMC_RUN_SMTLIB=1; shift ;;
    --bmc-assume-known-inputs)
      BMC_ASSUME_KNOWN_INPUTS=1; shift ;;
    --lec-assume-known-inputs)
      LEC_ASSUME_KNOWN_INPUTS=1; shift ;;
    --lec-accept-xprop-only)
      LEC_ACCEPT_XPROP_ONLY=1; shift ;;
    --update-baselines)
      UPDATE_BASELINES=1; shift ;;
    --fail-on-diff)
      FAIL_ON_DIFF=1; shift ;;
    --strict-gate)
      STRICT_GATE=1; shift ;;
    --baseline-window)
      BASELINE_WINDOW="$2"; shift 2 ;;
    --fail-on-new-xpass)
      FAIL_ON_NEW_XPASS=1; shift ;;
    --fail-on-passrate-regression)
      FAIL_ON_PASSRATE_REGRESSION=1; shift ;;
    --json-summary)
      JSON_SUMMARY_FILE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$PWD/formal-results-${DATE_STR//-/}"
fi
if ! [[ "$BASELINE_WINDOW" =~ ^[0-9]+$ ]] || [[ "$BASELINE_WINDOW" == "0" ]]; then
  echo "invalid --baseline-window: expected positive integer" >&2
  exit 1
fi

if [[ -z "$CIRCT_VERILOG_BIN_AVIP" ]]; then
  CIRCT_VERILOG_BIN_AVIP="$CIRCT_VERILOG_BIN"
fi
if [[ -z "$CIRCT_VERILOG_BIN_OPENTITAN" ]]; then
  CIRCT_VERILOG_BIN_OPENTITAN="$CIRCT_VERILOG_BIN"
fi

if [[ "$WITH_OPENTITAN" == "1" ]]; then
  if [[ ! -x "$CIRCT_VERILOG_BIN_OPENTITAN" ]]; then
    echo "circt-verilog for OpenTitan not executable: $CIRCT_VERILOG_BIN_OPENTITAN" >&2
    exit 1
  fi
fi
if [[ "$WITH_AVIP" == "1" ]]; then
  if [[ ! -x "$CIRCT_VERILOG_BIN_AVIP" ]]; then
    echo "circt-verilog for AVIP not executable: $CIRCT_VERILOG_BIN_AVIP" >&2
    exit 1
  fi
fi
if [[ "$STRICT_GATE" == "1" ]]; then
  FAIL_ON_NEW_XPASS=1
  FAIL_ON_PASSRATE_REGRESSION=1
fi
if [[ -z "$JSON_SUMMARY_FILE" ]]; then
  JSON_SUMMARY_FILE="$OUT_DIR/summary.json"
fi

mkdir -p "$OUT_DIR"

run_suite() {
  local name="$1"; shift
  local log="$OUT_DIR/${name}.log"
  local ec=0
  echo "==> ${name}" | tee "$log"
  set +e
  "$@" >>"$log" 2>&1
  ec=$?
  set -e
  echo "$ec" > "$OUT_DIR/${name}.exit"
  return $ec
}

extract_kv() {
  local line="$1"
  local key="$2"
  echo "$line" | tr ' ' '\n' | sed -n "s/^${key}=\([0-9]\+\)$/\\1/p"
}

results_tsv="$OUT_DIR/summary.tsv"
: > "$results_tsv"

printf "suite\tmode\ttotal\tpass\tfail\txfail\txpass\terror\tskip\tsummary\n" >> "$results_tsv"

record_result() {
  local suite="$1" mode="$2" total="$3" pass="$4" fail="$5" xfail="$6" xpass="$7" error="$8" skip="$9"
  local summary="total=${total} pass=${pass} fail=${fail} xfail=${xfail} xpass=${xpass} error=${error} skip=${skip}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$suite" "$mode" "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip" "$summary" >> "$results_tsv"
}

record_simple_result() {
  local suite="$1" mode="$2" exit_code="$3"
  local total=1 pass=0 fail=0 xfail=0 xpass=0 error=0 skip=0
  if [[ "$exit_code" == "0" ]]; then
    pass=1
  else
    fail=1
  fi
  record_result "$suite" "$mode" "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip"
}

# sv-tests BMC
if [[ -d "$SV_TESTS_DIR" ]]; then
  run_suite sv-tests-bmc \
    env OUT="$OUT_DIR/sv-tests-bmc-results.txt" \
    BMC_RUN_SMTLIB="$BMC_RUN_SMTLIB" \
    BMC_ASSUME_KNOWN_INPUTS="$BMC_ASSUME_KNOWN_INPUTS" \
    Z3_BIN="$Z3_BIN" \
    utils/run_sv_tests_circt_bmc.sh "$SV_TESTS_DIR" || true
  line=$(grep -E "sv-tests SVA summary:" "$OUT_DIR/sv-tests-bmc.log" | tail -1 || true)
  if [[ -n "$line" ]]; then
    total=$(extract_kv "$line" total)
    pass=$(extract_kv "$line" pass)
    fail=$(extract_kv "$line" fail)
    xfail=$(extract_kv "$line" xfail)
    xpass=$(extract_kv "$line" xpass)
    error=$(extract_kv "$line" error)
    skip=$(extract_kv "$line" skip)
    record_result "sv-tests" "BMC" "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip"
  fi
fi

# sv-tests LEC
if [[ -d "$SV_TESTS_DIR" ]]; then
  run_suite sv-tests-lec \
    env OUT="$OUT_DIR/sv-tests-lec-results.txt" \
    LEC_ASSUME_KNOWN_INPUTS="$LEC_ASSUME_KNOWN_INPUTS" \
    LEC_ACCEPT_XPROP_ONLY="$LEC_ACCEPT_XPROP_ONLY" \
    Z3_BIN="$Z3_BIN" \
    utils/run_sv_tests_circt_lec.sh "$SV_TESTS_DIR" || true
  line=$(grep -E "sv-tests LEC summary:" "$OUT_DIR/sv-tests-lec.log" | tail -1 || true)
  if [[ -n "$line" ]]; then
    total=$(extract_kv "$line" total)
    pass=$(extract_kv "$line" pass)
    fail=$(extract_kv "$line" fail)
    error=$(extract_kv "$line" error)
    skip=$(extract_kv "$line" skip)
    record_result "sv-tests" "LEC" "$total" "$pass" "$fail" 0 0 "$error" "$skip"
  fi
fi

# verilator-verification BMC
if [[ -d "$VERILATOR_DIR" ]]; then
  run_suite verilator-bmc \
    env OUT="$OUT_DIR/verilator-bmc-results.txt" \
    BMC_RUN_SMTLIB="$BMC_RUN_SMTLIB" \
    BMC_ASSUME_KNOWN_INPUTS="$BMC_ASSUME_KNOWN_INPUTS" \
    Z3_BIN="$Z3_BIN" \
    utils/run_verilator_verification_circt_bmc.sh "$VERILATOR_DIR" || true
  line=$(grep -E "verilator-verification summary:" "$OUT_DIR/verilator-bmc.log" | tail -1 || true)
  if [[ -n "$line" ]]; then
    total=$(extract_kv "$line" total)
    pass=$(extract_kv "$line" pass)
    fail=$(extract_kv "$line" fail)
    xfail=$(extract_kv "$line" xfail)
    xpass=$(extract_kv "$line" xpass)
    error=$(extract_kv "$line" error)
    skip=$(extract_kv "$line" skip)
    record_result "verilator-verification" "BMC" "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip"
  fi
fi

# verilator-verification LEC
if [[ -d "$VERILATOR_DIR" ]]; then
  run_suite verilator-lec \
    env OUT="$OUT_DIR/verilator-lec-results.txt" \
    LEC_ASSUME_KNOWN_INPUTS="$LEC_ASSUME_KNOWN_INPUTS" \
    LEC_ACCEPT_XPROP_ONLY="$LEC_ACCEPT_XPROP_ONLY" \
    Z3_BIN="$Z3_BIN" \
    utils/run_verilator_verification_circt_lec.sh "$VERILATOR_DIR" || true
  line=$(grep -E "verilator-verification LEC summary:" "$OUT_DIR/verilator-lec.log" | tail -1 || true)
  if [[ -n "$line" ]]; then
    total=$(extract_kv "$line" total)
    pass=$(extract_kv "$line" pass)
    fail=$(extract_kv "$line" fail)
    error=$(extract_kv "$line" error)
    skip=$(extract_kv "$line" skip)
    record_result "verilator-verification" "LEC" "$total" "$pass" "$fail" 0 0 "$error" "$skip"
  fi
fi

# yosys SVA BMC
if [[ -d "$YOSYS_DIR" ]]; then
  # NOTE: Do not pass BMC_ASSUME_KNOWN_INPUTS here; the yosys script defaults
  # it to 1 because yosys SVA tests are 2-state and need --assume-known-inputs
  # to avoid spurious X-driven counterexamples.  Only forward an explicit
  # override from the user (--bmc-assume-known-inputs flag).
  yosys_bmc_env=(OUT="$OUT_DIR/yosys-bmc-results.txt"
    BMC_RUN_SMTLIB="$BMC_RUN_SMTLIB"
    Z3_BIN="$Z3_BIN")
  if [[ "$BMC_ASSUME_KNOWN_INPUTS" == "1" ]]; then
    yosys_bmc_env+=(BMC_ASSUME_KNOWN_INPUTS=1)
  fi
  run_suite yosys-bmc \
    env "${yosys_bmc_env[@]}" \
    utils/run_yosys_sva_circt_bmc.sh "$YOSYS_DIR" || true
  line=$(grep -E "yosys SVA summary:" "$OUT_DIR/yosys-bmc.log" | tail -1 || true)
  if [[ -n "$line" ]]; then
    total=$(echo "$line" | sed -n 's/.*summary: \([0-9]\+\) tests.*/\1/p')
    failures=$(echo "$line" | sed -n 's/.*failures=\([0-9]\+\).*/\1/p')
    skipped=$(echo "$line" | sed -n 's/.*skipped=\([0-9]\+\).*/\1/p')
    pass=$((total - failures - skipped))
    record_result "yosys/tests/sva" "BMC" "$total" "$pass" "$failures" 0 0 0 "$skipped"
  fi
fi

# yosys SVA LEC
if [[ -d "$YOSYS_DIR" ]]; then
  run_suite yosys-lec \
    env OUT="$OUT_DIR/yosys-lec-results.txt" \
    LEC_ASSUME_KNOWN_INPUTS="$LEC_ASSUME_KNOWN_INPUTS" \
    LEC_ACCEPT_XPROP_ONLY="$LEC_ACCEPT_XPROP_ONLY" \
    Z3_BIN="$Z3_BIN" \
    utils/run_yosys_sva_circt_lec.sh "$YOSYS_DIR" || true
  line=$(grep -E "yosys LEC summary:" "$OUT_DIR/yosys-lec.log" | tail -1 || true)
  if [[ -n "$line" ]]; then
    total=$(extract_kv "$line" total)
    pass=$(extract_kv "$line" pass)
    fail=$(extract_kv "$line" fail)
    error=$(extract_kv "$line" error)
    skip=$(extract_kv "$line" skip)
    record_result "yosys/tests/sva" "LEC" "$total" "$pass" "$fail" 0 0 "$error" "$skip"
  fi
fi

# OpenTitan LEC (optional)
if [[ "$WITH_OPENTITAN" == "1" ]]; then
  opentitan_env=(LEC_ASSUME_KNOWN_INPUTS="$LEC_ASSUME_KNOWN_INPUTS"
    CIRCT_VERILOG="$CIRCT_VERILOG_BIN_OPENTITAN")
  if [[ "$LEC_ACCEPT_XPROP_ONLY" == "1" ]]; then
    opentitan_env+=(CIRCT_LEC_ARGS="--accept-xprop-only ${CIRCT_LEC_ARGS:-}")
  fi
  run_suite opentitan-lec \
    env "${opentitan_env[@]}" \
    utils/run_opentitan_circt_lec.py --opentitan-root "$OPENTITAN_DIR" || true
  line=$(grep -E "Running LEC on [0-9]+" "$OUT_DIR/opentitan-lec.log" | tail -1 || true)
  total=$(echo "$line" | sed -n 's/.*Running LEC on \([0-9]\+\).*/\1/p')
  failures=$(grep -E "LEC failures: [0-9]+" "$OUT_DIR/opentitan-lec.log" | tail -1 | sed -n 's/.*LEC failures: \([0-9]\+\).*/\1/p' || true)
  if [[ -n "$total" ]]; then
    if [[ -z "$failures" ]]; then
      failures=0
    fi
    pass=$((total - failures))
    record_result "opentitan" "LEC" "$total" "$pass" "$failures" 0 0 0 0
  fi
fi

# AVIP compile smoke (optional)
if [[ "$WITH_AVIP" == "1" ]]; then
  for avip in $AVIP_GLOB; do
    if [[ -d "$avip" ]]; then
      avip_name="$(basename "$avip")"
      run_suite "avip-${avip_name}" \
        env OUT="$OUT_DIR/${avip_name}-circt-verilog.log" \
        CIRCT_VERILOG="$CIRCT_VERILOG_BIN_AVIP" \
        utils/run_avip_circt_verilog.sh "$avip" || true
      if [[ -f "$OUT_DIR/avip-${avip_name}.exit" ]]; then
        avip_ec=$(cat "$OUT_DIR/avip-${avip_name}.exit")
        record_simple_result "avip/${avip_name}" "compile" "$avip_ec"
      fi
    fi
  done
fi

summary_txt="$OUT_DIR/summary.txt"
{
  echo "Formal suite summary (${DATE_STR})"
  printf "%-28s %-6s %-6s %s\n" "Suite" "Mode" "Status" "Details"
  echo "---------------------------------------------------------------"
  tail -n +2 "$results_tsv" | while IFS=$'\t' read -r suite mode total pass fail xfail xpass error skip summary; do
    status="PASS"
    if [[ "$mode" == "BMC" ]]; then
      if [[ "$fail" != "0" || "$error" != "0" || "$xpass" != "0" ]]; then
        status="FAIL"
      fi
    else
      if [[ "$fail" != "0" || "$error" != "0" ]]; then
        status="FAIL"
      fi
    fi
    printf "%-28s %-6s %-6s %s\n" "$suite" "$mode" "$status" "$summary"
  done
  echo "Logs: $OUT_DIR"
} | tee "$summary_txt"

OUT_DIR="$OUT_DIR" JSON_SUMMARY_FILE="$JSON_SUMMARY_FILE" DATE_STR="$DATE_STR" python3 - <<'PY'
import csv
import json
import os
import subprocess
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
summary_path = out_dir / "summary.tsv"
json_summary_path = Path(os.environ["JSON_SUMMARY_FILE"])

try:
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
except Exception:
    git_sha = ""

rows = []
with summary_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        total = int(row["total"])
        passed = int(row["pass"])
        xfail = int(row["xfail"])
        skipped = int(row["skip"])
        eligible = max(total - skipped, 0)
        pass_rate = 0.0
        if eligible > 0:
            pass_rate = ((passed + xfail) * 100.0) / eligible
        rows.append(
            {
                "suite": row["suite"],
                "mode": row["mode"],
                "total": total,
                "pass": passed,
                "fail": int(row["fail"]),
                "xfail": xfail,
                "xpass": int(row["xpass"]),
                "error": int(row["error"]),
                "skip": skipped,
                "pass_rate": round(pass_rate, 3),
                "summary": row["summary"],
            }
        )

payload = {
    "date": os.environ.get("DATE_STR", ""),
    "git_sha": git_sha,
    "rows": rows,
}
json_summary_path.parent.mkdir(parents=True, exist_ok=True)
json_summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

if [[ "$UPDATE_BASELINES" == "1" ]]; then
  OUT_DIR="$OUT_DIR" DATE_STR="$DATE_STR" BASELINE_FILE="$BASELINE_FILE" PLAN_FILE="$PLAN_FILE" python3 - <<'PY'
import csv
import os
import re
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
summary_path = out_dir / "summary.tsv"
baseline_path = Path(os.environ["BASELINE_FILE"])
plan_path = Path(os.environ["PLAN_FILE"])

date_str = os.environ["DATE_STR"]

rows = []
with summary_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        rows.append(row)

def parse_result_summary(summary: str):
    parsed = {}
    for match in re.finditer(r"([a-z_]+)=([0-9]+)", summary):
        parsed[match.group(1)] = int(match.group(2))
    return parsed

def read_baseline_int(row, key, summary_counts):
    raw = row.get(key)
    if raw is not None and raw != "":
        try:
            return int(raw)
        except ValueError:
            pass
    return int(summary_counts.get(key, 0))

def read_int(row, key, fallback=0):
    value = row.get(key)
    if value is None or value == "":
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback

def compute_pass_rate(total: int, passed: int, xfail: int, skipped: int) -> float:
    eligible = total - skipped
    if eligible <= 0:
        return 0.0
    return ((passed + xfail) * 100.0) / eligible

baseline = {}
if baseline_path.exists():
    with baseline_path.open() as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            key = (row['date'], row['suite'], row['mode'])
            summary_counts = parse_result_summary(row.get('result', ''))
            total = read_int(row, 'total', summary_counts.get('total', 0))
            passed = read_int(row, 'pass', summary_counts.get('pass', 0))
            fail = read_int(row, 'fail', summary_counts.get('fail', 0))
            xfail = read_int(row, 'xfail', summary_counts.get('xfail', 0))
            xpass = read_int(row, 'xpass', summary_counts.get('xpass', 0))
            error = read_int(row, 'error', summary_counts.get('error', 0))
            skip = read_int(row, 'skip', summary_counts.get('skip', 0))
            pass_rate = row.get('pass_rate', '')
            if not pass_rate:
                pass_rate = f"{compute_pass_rate(total, passed, xfail, skip):.3f}"
            baseline[key] = {
                'date': row['date'],
                'suite': row['suite'],
                'mode': row['mode'],
                'total': str(total),
                'pass': str(passed),
                'fail': str(fail),
                'xfail': str(xfail),
                'xpass': str(xpass),
                'error': str(error),
                'skip': str(skip),
                'pass_rate': pass_rate,
                'result': row.get('result', ''),
            }

for row in rows:
    key = (date_str, row['suite'], row['mode'])
    total = int(row['total'])
    passed = int(row['pass'])
    fail = int(row['fail'])
    xfail = int(row['xfail'])
    xpass = int(row['xpass'])
    error = int(row['error'])
    skip = int(row['skip'])
    pass_rate = compute_pass_rate(total, passed, xfail, skip)
    baseline[key] = {
        'date': date_str,
        'suite': row['suite'],
        'mode': row['mode'],
        'total': str(total),
        'pass': str(passed),
        'fail': str(fail),
        'xfail': str(xfail),
        'xpass': str(xpass),
        'error': str(error),
        'skip': str(skip),
        'pass_rate': f"{pass_rate:.3f}",
        'result': row['summary'],
    }

baseline_path.parent.mkdir(parents=True, exist_ok=True)
with baseline_path.open('w', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            'date',
            'suite',
            'mode',
            'total',
            'pass',
            'fail',
            'xfail',
            'xpass',
            'error',
            'skip',
            'pass_rate',
            'result',
        ],
        delimiter='\t',
    )
    writer.writeheader()
    for key in sorted(baseline.keys()):
        writer.writerow(baseline[key])

if plan_path.exists():
    data = plan_path.read_text().splitlines()
    header_idx = None
    table_end = None
    for idx, line in enumerate(data):
        if line.strip() == "| Date | Suite | Mode | Result | Notes |":
            header_idx = idx
            continue
        if header_idx is not None and idx > header_idx:
            if line.strip().startswith("|"):
                table_end = idx
                continue
            table_end = idx - 1
            break
    if header_idx is not None:
        if table_end is None:
            table_end = header_idx
        seen = set()
        for idx in range(header_idx + 1, table_end + 1):
            line = data[idx]
            parts = [p.strip() for p in line.strip().strip('|').split('|')]
            if len(parts) < 5:
                continue
            date, suite, mode, result, notes = parts[:5]
            key = (date, suite, mode)
            seen.add(key)
            if key in baseline:
                result = baseline[key]['result']
                data[idx] = f"| {date} | {suite} | {mode} | {result} | {notes} |"
        new_rows = []
        for key, row in sorted(baseline.items()):
            if row['date'] != date_str:
                continue
            if key in seen:
                continue
            new_rows.append(
                f"| {row['date']} | {row['suite']} | {row['mode']} | {row['result']} | added by script |"
            )
        if new_rows:
            insert_at = table_end + 1
            data[insert_at:insert_at] = new_rows
        plan_path.write_text("\n".join(data) + "\n")
PY
fi

if [[ "$FAIL_ON_NEW_XPASS" == "1" || "$FAIL_ON_PASSRATE_REGRESSION" == "1" ]]; then
  OUT_DIR="$OUT_DIR" BASELINE_FILE="$BASELINE_FILE" \
  BASELINE_WINDOW="$BASELINE_WINDOW" \
  FAIL_ON_NEW_XPASS="$FAIL_ON_NEW_XPASS" \
  FAIL_ON_PASSRATE_REGRESSION="$FAIL_ON_PASSRATE_REGRESSION" \
  STRICT_GATE="$STRICT_GATE" python3 - <<'PY'
import csv
import os
import re
from pathlib import Path

summary_path = Path(os.environ["OUT_DIR"]) / "summary.tsv"
baseline_path = Path(os.environ["BASELINE_FILE"])

if not baseline_path.exists():
    raise SystemExit(f"baseline file not found: {baseline_path}")

def parse_result_summary(summary: str):
    parsed = {}
    for match in re.finditer(r"([a-z_]+)=([0-9]+)", summary):
        parsed[match.group(1)] = int(match.group(2))
    return parsed

def read_baseline_int(row, key, summary_counts):
    raw = row.get(key)
    if raw is not None and raw != "":
        try:
            return int(raw)
        except ValueError:
            pass
    return int(summary_counts.get(key, 0))

def pass_rate(row):
    total = int(row.get("total", 0))
    passed = int(row.get("pass", 0))
    xfail = int(row.get("xfail", 0))
    skipped = int(row.get("skip", 0))
    eligible = total - skipped
    if eligible <= 0:
        return 0.0
    return ((passed + xfail) * 100.0) / eligible

summary = {}
with summary_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        key = (row["suite"], row["mode"])
        summary[key] = row

history = {}
with baseline_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        suite = row.get("suite", "")
        mode = row.get("mode", "")
        if not suite or not mode:
            continue
        key = (suite, mode)
        history.setdefault(key, []).append(row)

fail_on_new_xpass = os.environ.get("FAIL_ON_NEW_XPASS", "0") == "1"
fail_on_passrate_regression = os.environ.get("FAIL_ON_PASSRATE_REGRESSION", "0") == "1"
strict_gate = os.environ.get("STRICT_GATE", "0") == "1"
baseline_window = int(os.environ.get("BASELINE_WINDOW", "1"))

gate_errors = []
for key, current_row in summary.items():
    suite, mode = key
    history_rows = history.get(key, [])
    if not history_rows:
        if strict_gate:
            gate_errors.append(f"{suite} {mode}: missing baseline row")
        continue
    history_rows.sort(key=lambda r: r.get("date", ""))
    if strict_gate and len(history_rows) < baseline_window:
        gate_errors.append(
            f"{suite} {mode}: insufficient baseline history ({len(history_rows)} < {baseline_window})"
        )
        continue
    compare_rows = history_rows[-baseline_window:]
    parsed_counts = [parse_result_summary(row.get("result", "")) for row in compare_rows]
    baseline_fail = min(
        read_baseline_int(row, "fail", counts)
        for row, counts in zip(compare_rows, parsed_counts)
    )
    baseline_error = min(
        read_baseline_int(row, "error", counts)
        for row, counts in zip(compare_rows, parsed_counts)
    )
    baseline_xpass = min(
        read_baseline_int(row, "xpass", counts)
        for row, counts in zip(compare_rows, parsed_counts)
    )
    current_fail = int(current_row["fail"])
    current_error = int(current_row["error"])
    current_xpass = int(current_row["xpass"])
    if current_fail > baseline_fail:
        gate_errors.append(
            f"{suite} {mode}: fail increased ({baseline_fail} -> {current_fail}, window={baseline_window})"
        )
    if current_error > baseline_error:
        gate_errors.append(
            f"{suite} {mode}: error increased ({baseline_error} -> {current_error}, window={baseline_window})"
        )
    if fail_on_new_xpass and current_xpass > baseline_xpass:
        gate_errors.append(
            f"{suite} {mode}: xpass increased ({baseline_xpass} -> {current_xpass}, window={baseline_window})"
        )
    if fail_on_passrate_regression:
        baseline_rate = max(
            pass_rate(
                {
                    "total": read_baseline_int(row, "total", counts),
                    "pass": read_baseline_int(row, "pass", counts),
                    "xfail": read_baseline_int(row, "xfail", counts),
                    "skip": read_baseline_int(row, "skip", counts),
                }
            )
            for row, counts in zip(compare_rows, parsed_counts)
        )
        current_rate = pass_rate(current_row)
        if current_rate + 1e-9 < baseline_rate:
            gate_errors.append(
                f"{suite} {mode}: pass_rate regressed ({baseline_rate:.3f} -> {current_rate:.3f}, window={baseline_window})"
            )

if gate_errors:
    print("strict gate failures:")
    for item in gate_errors:
        print(f"  {item}")
    raise SystemExit(1)
PY
fi

if [[ "$FAIL_ON_DIFF" == "1" ]]; then
  OUT_DIR="$OUT_DIR" BASELINE_FILE="$BASELINE_FILE" python3 - <<'PY'
import csv
import os
from pathlib import Path

summary_path = Path(os.environ["OUT_DIR"]) / "summary.tsv"
baseline_path = Path(os.environ["BASELINE_FILE"])

if not baseline_path.exists():
    raise SystemExit("baseline file not found: utils/formal-baselines.tsv")

summary = {}
with summary_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        key = (row['suite'], row['mode'])
        summary[key] = row['summary']

latest = {}
with baseline_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        key = (row['suite'], row['mode'])
        latest.setdefault(key, []).append(row)

for key, entries in latest.items():
    entries.sort(key=lambda r: r['date'])
    latest[key] = entries[-1]['result']

diffs = []
for key, summary_val in summary.items():
    baseline_val = latest.get(key)
    if baseline_val is None:
        diffs.append(f"missing baseline for {key[0]} {key[1]}")
    elif baseline_val != summary_val:
        diffs.append(f"{key[0]} {key[1]}: {baseline_val} -> {summary_val}")

if diffs:
    print("baseline diffs:")
    for item in diffs:
        print(f"  {item}")
    raise SystemExit(1)
PY
fi

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
  --baseline-window-days N
                         Limit baseline comparison to rows within N days of the
                         suite/mode latest baseline date (default: 0, disabled)
  --fail-on-new-xpass    Fail when xpass count increases vs baseline
  --fail-on-passrate-regression
                         Fail when pass-rate decreases vs baseline
  --expected-failures-file FILE
                         TSV with suite/mode expected fail+error budgets
  --expectations-dry-run
                         Preview expectation refresh/prune without rewriting files
  --expectations-dry-run-report-jsonl FILE
                         Append dry-run operation summaries as JSON Lines
  --expectations-dry-run-report-max-sample-rows N
                         Max sampled rows embedded per dry-run JSONL operation
                         (default: 5; 0 disables row samples)
  --expectations-dry-run-report-hmac-key-file FILE
                         Optional key file for HMAC-SHA256 signing of run_end
                         payload digest
  --expectations-dry-run-report-hmac-key-id ID
                         Optional key identifier emitted in run_meta/run_end
                         when HMAC signing is enabled
  --fail-on-unexpected-failures
                         Fail when fail/error exceed expected-failure budgets
  --fail-on-unused-expected-failures
                         Fail when expected-failures rows are unused
  --prune-expected-failures-file FILE
                         Rewrite expected-failures file by pruning stale rows
  --prune-expected-failures-drop-unused
                         Drop expected-failures rows unused in current run
  --refresh-expected-failures-file FILE
                         Rewrite expected-failures TSV from current run
  --refresh-expected-failures-include-suite-regex REGEX
                         Refresh only suite rows matching REGEX
  --refresh-expected-failures-include-mode-regex REGEX
                         Refresh only mode rows matching REGEX
  --expected-failure-cases-file FILE
                         TSV with expected failing test cases (suite/mode/id)
  --fail-on-unexpected-failure-cases
                         Fail when observed failing cases are not expected
  --fail-on-expired-expected-failure-cases
                         Fail when any expected case is expired by expires_on
  --fail-on-unmatched-expected-failure-cases
                         Fail when expected cases have no observed match
  --prune-expected-failure-cases-file FILE
                         Rewrite expected-cases file by pruning stale rows
  --prune-expected-failure-cases-drop-unmatched
                         Drop expected-case rows with matched_count=0
  --prune-expected-failure-cases-drop-expired
                         Drop expected-case rows with expired=yes
  --refresh-expected-failure-cases-file FILE
                         Rewrite expected-failure-cases TSV from current run
  --refresh-expected-failure-cases-default-expires-on YYYY-MM-DD
                         Default expires_on for newly added refreshed case rows
  --refresh-expected-failure-cases-collapse-status-any
                         Collapse refreshed case statuses to ANY per case key
  --refresh-expected-failure-cases-include-suite-regex REGEX
                         Refresh only case rows with suite matching REGEX
  --refresh-expected-failure-cases-include-mode-regex REGEX
                         Refresh only case rows with mode matching REGEX
  --refresh-expected-failure-cases-include-status-regex REGEX
                         Refresh only case rows with status matching REGEX
  --refresh-expected-failure-cases-include-id-regex REGEX
                         Refresh only case rows with id matching REGEX
  --json-summary FILE    Write machine-readable JSON summary (default: <out-dir>/summary.json)
  --include-lane-regex REGEX
                         Run only lanes whose lane-id matches REGEX
  --exclude-lane-regex REGEX
                         Skip lanes whose lane-id matches REGEX
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
RUN_ID="${DATE_STR}-$$-$(date +%H%M%S)"
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
BASELINE_WINDOW_DAYS=0
FAIL_ON_NEW_XPASS=0
FAIL_ON_PASSRATE_REGRESSION=0
EXPECTED_FAILURES_FILE=""
EXPECTATIONS_DRY_RUN=0
EXPECTATIONS_DRY_RUN_REPORT_JSONL=""
EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS=5
EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE=""
EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID=""
FAIL_ON_UNEXPECTED_FAILURES=0
FAIL_ON_UNUSED_EXPECTED_FAILURES=0
PRUNE_EXPECTED_FAILURES_FILE=""
PRUNE_EXPECTED_FAILURES_DROP_UNUSED=0
REFRESH_EXPECTED_FAILURES_FILE=""
REFRESH_EXPECTED_FAILURES_INCLUDE_SUITE_REGEX=""
REFRESH_EXPECTED_FAILURES_INCLUDE_MODE_REGEX=""
EXPECTED_FAILURE_CASES_FILE=""
FAIL_ON_UNEXPECTED_FAILURE_CASES=0
FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES=0
FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES=0
PRUNE_EXPECTED_FAILURE_CASES_FILE=""
PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED=0
PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED=0
REFRESH_EXPECTED_FAILURE_CASES_FILE=""
REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON=""
REFRESH_EXPECTED_FAILURE_CASES_COLLAPSE_STATUS_ANY=0
REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_SUITE_REGEX=""
REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_MODE_REGEX=""
REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_STATUS_REGEX=""
REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_ID_REGEX=""
JSON_SUMMARY_FILE=""
INCLUDE_LANE_REGEX=""
EXCLUDE_LANE_REGEX=""
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
    --baseline-window-days)
      BASELINE_WINDOW_DAYS="$2"; shift 2 ;;
    --fail-on-new-xpass)
      FAIL_ON_NEW_XPASS=1; shift ;;
    --fail-on-passrate-regression)
      FAIL_ON_PASSRATE_REGRESSION=1; shift ;;
    --expected-failures-file)
      EXPECTED_FAILURES_FILE="$2"; shift 2 ;;
    --expectations-dry-run)
      EXPECTATIONS_DRY_RUN=1; shift ;;
    --expectations-dry-run-report-jsonl)
      EXPECTATIONS_DRY_RUN_REPORT_JSONL="$2"; shift 2 ;;
    --expectations-dry-run-report-max-sample-rows)
      EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS="$2"; shift 2 ;;
    --expectations-dry-run-report-hmac-key-file)
      EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE="$2"; shift 2 ;;
    --expectations-dry-run-report-hmac-key-id)
      EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID="$2"; shift 2 ;;
    --fail-on-unexpected-failures)
      FAIL_ON_UNEXPECTED_FAILURES=1; shift ;;
    --fail-on-unused-expected-failures)
      FAIL_ON_UNUSED_EXPECTED_FAILURES=1; shift ;;
    --prune-expected-failures-file)
      PRUNE_EXPECTED_FAILURES_FILE="$2"; shift 2 ;;
    --prune-expected-failures-drop-unused)
      PRUNE_EXPECTED_FAILURES_DROP_UNUSED=1; shift ;;
    --refresh-expected-failures-file)
      REFRESH_EXPECTED_FAILURES_FILE="$2"; shift 2 ;;
    --refresh-expected-failures-include-suite-regex)
      REFRESH_EXPECTED_FAILURES_INCLUDE_SUITE_REGEX="$2"; shift 2 ;;
    --refresh-expected-failures-include-mode-regex)
      REFRESH_EXPECTED_FAILURES_INCLUDE_MODE_REGEX="$2"; shift 2 ;;
    --expected-failure-cases-file)
      EXPECTED_FAILURE_CASES_FILE="$2"; shift 2 ;;
    --fail-on-unexpected-failure-cases)
      FAIL_ON_UNEXPECTED_FAILURE_CASES=1; shift ;;
    --fail-on-expired-expected-failure-cases)
      FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES=1; shift ;;
    --fail-on-unmatched-expected-failure-cases)
      FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES=1; shift ;;
    --prune-expected-failure-cases-file)
      PRUNE_EXPECTED_FAILURE_CASES_FILE="$2"; shift 2 ;;
    --prune-expected-failure-cases-drop-unmatched)
      PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED=1; shift ;;
    --prune-expected-failure-cases-drop-expired)
      PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED=1; shift ;;
    --refresh-expected-failure-cases-file)
      REFRESH_EXPECTED_FAILURE_CASES_FILE="$2"; shift 2 ;;
    --refresh-expected-failure-cases-default-expires-on)
      REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON="$2"; shift 2 ;;
    --refresh-expected-failure-cases-collapse-status-any)
      REFRESH_EXPECTED_FAILURE_CASES_COLLAPSE_STATUS_ANY=1; shift ;;
    --refresh-expected-failure-cases-include-suite-regex)
      REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_SUITE_REGEX="$2"; shift 2 ;;
    --refresh-expected-failure-cases-include-mode-regex)
      REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_MODE_REGEX="$2"; shift 2 ;;
    --refresh-expected-failure-cases-include-status-regex)
      REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_STATUS_REGEX="$2"; shift 2 ;;
    --refresh-expected-failure-cases-include-id-regex)
      REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_ID_REGEX="$2"; shift 2 ;;
    --json-summary)
      JSON_SUMMARY_FILE="$2"; shift 2 ;;
    --include-lane-regex)
      INCLUDE_LANE_REGEX="$2"; shift 2 ;;
    --exclude-lane-regex)
      EXCLUDE_LANE_REGEX="$2"; shift 2 ;;
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
if ! [[ "$BASELINE_WINDOW_DAYS" =~ ^[0-9]+$ ]]; then
  echo "invalid --baseline-window-days: expected non-negative integer" >&2
  exit 1
fi
if ! [[ "$EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS" =~ ^[0-9]+$ ]]; then
  echo "invalid --expectations-dry-run-report-max-sample-rows: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$INCLUDE_LANE_REGEX" ]]; then
  set +e
  printf '' | grep -Eq "$INCLUDE_LANE_REGEX" 2>/dev/null
  lane_regex_ec=$?
  set -e
  if [[ "$lane_regex_ec" == "2" ]]; then
    echo "invalid --include-lane-regex: $INCLUDE_LANE_REGEX" >&2
    exit 1
  fi
fi
if [[ -n "$EXCLUDE_LANE_REGEX" ]]; then
  set +e
  printf '' | grep -Eq "$EXCLUDE_LANE_REGEX" 2>/dev/null
  lane_regex_ec=$?
  set -e
  if [[ "$lane_regex_ec" == "2" ]]; then
    echo "invalid --exclude-lane-regex: $EXCLUDE_LANE_REGEX" >&2
    exit 1
  fi
fi
if [[ -n "$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE" && \
      ! -r "$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE" ]]; then
  echo "expectations dry-run report HMAC key file not readable: $EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE" >&2
  exit 1
fi
if [[ -n "$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID" && \
      -z "$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE" ]]; then
  echo "--expectations-dry-run-report-hmac-key-id requires --expectations-dry-run-report-hmac-key-file" >&2
  exit 1
fi
if [[ "$FAIL_ON_UNEXPECTED_FAILURES" == "1" && -z "$EXPECTED_FAILURES_FILE" ]]; then
  echo "--fail-on-unexpected-failures requires --expected-failures-file" >&2
  exit 1
fi
if [[ "$FAIL_ON_UNUSED_EXPECTED_FAILURES" == "1" && -z "$EXPECTED_FAILURES_FILE" ]]; then
  echo "--fail-on-unused-expected-failures requires --expected-failures-file" >&2
  exit 1
fi
if [[ -n "$PRUNE_EXPECTED_FAILURES_FILE" && -z "$EXPECTED_FAILURES_FILE" ]]; then
  EXPECTED_FAILURES_FILE="$PRUNE_EXPECTED_FAILURES_FILE"
fi
if [[ -n "$PRUNE_EXPECTED_FAILURES_FILE" && "$PRUNE_EXPECTED_FAILURES_DROP_UNUSED" != "1" ]]; then
  PRUNE_EXPECTED_FAILURES_DROP_UNUSED=1
fi
if [[ -n "$PRUNE_EXPECTED_FAILURES_FILE" && ! -r "$PRUNE_EXPECTED_FAILURES_FILE" ]]; then
  echo "prune expected-failures file not readable: $PRUNE_EXPECTED_FAILURES_FILE" >&2
  exit 1
fi
if [[ -n "$EXPECTED_FAILURES_FILE" && ! -r "$EXPECTED_FAILURES_FILE" ]]; then
  echo "expected-failures file not readable: $EXPECTED_FAILURES_FILE" >&2
  exit 1
fi
if [[ "$FAIL_ON_UNEXPECTED_FAILURE_CASES" == "1" && -z "$EXPECTED_FAILURE_CASES_FILE" ]]; then
  echo "--fail-on-unexpected-failure-cases requires --expected-failure-cases-file" >&2
  exit 1
fi
if [[ "$FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES" == "1" && -z "$EXPECTED_FAILURE_CASES_FILE" ]]; then
  echo "--fail-on-expired-expected-failure-cases requires --expected-failure-cases-file" >&2
  exit 1
fi
if [[ "$FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES" == "1" && -z "$EXPECTED_FAILURE_CASES_FILE" ]]; then
  echo "--fail-on-unmatched-expected-failure-cases requires --expected-failure-cases-file" >&2
  exit 1
fi
if [[ -n "$PRUNE_EXPECTED_FAILURE_CASES_FILE" && -z "$EXPECTED_FAILURE_CASES_FILE" ]]; then
  EXPECTED_FAILURE_CASES_FILE="$PRUNE_EXPECTED_FAILURE_CASES_FILE"
fi
if [[ -n "$PRUNE_EXPECTED_FAILURE_CASES_FILE" && \
      "$PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED" != "1" && \
      "$PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED" != "1" ]]; then
  PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED=1
  PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED=1
fi
if [[ -n "$PRUNE_EXPECTED_FAILURE_CASES_FILE" && ! -r "$PRUNE_EXPECTED_FAILURE_CASES_FILE" ]]; then
  echo "prune expected-failure-cases file not readable: $PRUNE_EXPECTED_FAILURE_CASES_FILE" >&2
  exit 1
fi
if [[ -n "$EXPECTED_FAILURE_CASES_FILE" && ! -r "$EXPECTED_FAILURE_CASES_FILE" ]]; then
  echo "expected-failure-cases file not readable: $EXPECTED_FAILURE_CASES_FILE" >&2
  exit 1
fi
if [[ -n "$REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON" ]]; then
  if ! [[ "$REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "invalid --refresh-expected-failure-cases-default-expires-on: expected YYYY-MM-DD" >&2
    exit 1
  fi
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

emit_expectations_dry_run_run_end() {
  local exit_code="$?"
  if [[ "$EXPECTATIONS_DRY_RUN" == "1" && -n "$EXPECTATIONS_DRY_RUN_REPORT_JSONL" ]]; then
    OUT_DIR="$OUT_DIR" \
    DATE_STR="$DATE_STR" \
    EXPECTATIONS_DRY_RUN_RUN_ID="$RUN_ID" \
    EXPECTATIONS_DRY_RUN_REPORT_JSONL="$EXPECTATIONS_DRY_RUN_REPORT_JSONL" \
    EXPECTATIONS_DRY_RUN_EXIT_CODE="$exit_code" \
    EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE="$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE" \
    EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID="$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID" \
    python3 - <<'PY'
import hmac
import hashlib
import json
import os
from pathlib import Path

report_path = Path(os.environ["EXPECTATIONS_DRY_RUN_REPORT_JSONL"])
report_path.parent.mkdir(parents=True, exist_ok=True)
run_id = os.environ.get("EXPECTATIONS_DRY_RUN_RUN_ID", "")
rows_for_run = []
if report_path.exists():
  for line in report_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      row = json.loads(line)
    except Exception:
      continue
    if row.get("run_id", "") == run_id:
      rows_for_run.append(row)

digest = hashlib.sha256()
for row in rows_for_run:
  digest.update(
      json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
  )
  digest.update(b"\n")

payload_hash_hex = digest.hexdigest()
payload = {
    "operation": "run_end",
    "schema_version": 1,
    "date": os.environ.get("DATE_STR", ""),
    "run_id": run_id,
    "out_dir": os.environ.get("OUT_DIR", ""),
    "exit_code": int(os.environ.get("EXPECTATIONS_DRY_RUN_EXIT_CODE", "0")),
    "row_count": len(rows_for_run),
    "payload_sha256": payload_hash_hex,
}
hmac_key_file = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE", "")
if hmac_key_file:
  key_bytes = Path(hmac_key_file).read_bytes()
  key_id = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID", "").strip()
  payload["payload_hmac_sha256"] = hmac.new(
      key_bytes, payload_hash_hex.encode("utf-8"), hashlib.sha256
  ).hexdigest()
  if key_id:
    payload["hmac_key_id"] = key_id
with report_path.open("a", encoding="utf-8") as f:
  f.write(json.dumps(payload, sort_keys=True) + "\n")
PY
  fi
  return "$exit_code"
}
trap emit_expectations_dry_run_run_end EXIT

if [[ "$EXPECTATIONS_DRY_RUN" == "1" && -n "$EXPECTATIONS_DRY_RUN_REPORT_JSONL" ]]; then
  OUT_DIR="$OUT_DIR" \
  DATE_STR="$DATE_STR" \
  EXPECTATIONS_DRY_RUN_RUN_ID="$RUN_ID" \
  EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS="$EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS" \
  EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE="$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE" \
  EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID="$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID" \
  EXPECTATIONS_DRY_RUN_REPORT_JSONL="$EXPECTATIONS_DRY_RUN_REPORT_JSONL" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

report_path = Path(os.environ["EXPECTATIONS_DRY_RUN_REPORT_JSONL"])
report_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "operation": "run_meta",
    "schema_version": 1,
    "date": os.environ.get("DATE_STR", ""),
    "run_id": os.environ.get("EXPECTATIONS_DRY_RUN_RUN_ID", ""),
    "report_sample_rows_limit": int(
        os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS", "5")
    ),
    "hmac_mode": "sha256-keyfile"
    if os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE", "")
    else "none",
    "out_dir": os.environ.get("OUT_DIR", ""),
}
if payload["hmac_mode"] != "none":
  hmac_key_id = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID", "").strip()
  if hmac_key_id:
    payload["hmac_key_id"] = hmac_key_id
with report_path.open("a", encoding="utf-8") as f:
  f.write(json.dumps(payload, sort_keys=True) + "\n")
PY
fi

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

lane_enabled() {
  local lane_id="$1"
  if [[ -n "$INCLUDE_LANE_REGEX" ]]; then
    if ! printf '%s\n' "$lane_id" | grep -Eq "$INCLUDE_LANE_REGEX"; then
      return 1
    fi
  fi
  if [[ -n "$EXCLUDE_LANE_REGEX" ]]; then
    if printf '%s\n' "$lane_id" | grep -Eq "$EXCLUDE_LANE_REGEX"; then
      return 1
    fi
  fi
  return 0
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
if [[ -d "$SV_TESTS_DIR" ]] && lane_enabled "sv-tests/BMC"; then
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
if [[ -d "$SV_TESTS_DIR" ]] && lane_enabled "sv-tests/LEC"; then
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
if [[ -d "$VERILATOR_DIR" ]] && lane_enabled "verilator-verification/BMC"; then
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
if [[ -d "$VERILATOR_DIR" ]] && lane_enabled "verilator-verification/LEC"; then
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
if [[ -d "$YOSYS_DIR" ]] && lane_enabled "yosys/tests/sva/BMC"; then
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
if [[ -d "$YOSYS_DIR" ]] && lane_enabled "yosys/tests/sva/LEC"; then
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
if [[ "$WITH_OPENTITAN" == "1" ]] && lane_enabled "opentitan/LEC"; then
  opentitan_case_results="$OUT_DIR/opentitan-lec-results.txt"
  : > "$opentitan_case_results"
  opentitan_env=(LEC_ASSUME_KNOWN_INPUTS="$LEC_ASSUME_KNOWN_INPUTS"
    OUT="$opentitan_case_results"
    CIRCT_VERILOG="$CIRCT_VERILOG_BIN_OPENTITAN")
  if [[ "$LEC_ACCEPT_XPROP_ONLY" == "1" ]]; then
    opentitan_env+=(CIRCT_LEC_ARGS="--accept-xprop-only ${CIRCT_LEC_ARGS:-}")
  fi
  run_suite opentitan-lec \
    env "${opentitan_env[@]}" \
    utils/run_opentitan_circt_lec.py --opentitan-root "$OPENTITAN_DIR" || true
  if [[ ! -s "$opentitan_case_results" ]]; then
    OPENTITAN_LOG_FILE="$OUT_DIR/opentitan-lec.log" \
    OPENTITAN_CASE_RESULTS_FILE="$opentitan_case_results" python3 - <<'PY'
import os
import re
from pathlib import Path

log_path = Path(os.environ["OPENTITAN_LOG_FILE"])
out_path = Path(os.environ["OPENTITAN_CASE_RESULTS_FILE"])

rows = []
if log_path.exists():
  for line in log_path.read_text().splitlines():
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s+OK\s*$", line)
    if m:
      impl = m.group(1)
      rows.append(("PASS", impl, impl, "opentitan", "LEC"))
      continue
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s+XPROP_ONLY\s+\(accepted\)\s*$", line)
    if m:
      impl = m.group(1)
      rows.append(("XFAIL", impl, impl, "opentitan", "LEC"))
      continue
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s+FAIL(?:\s+\([^)]+\))?(?:\s+\(logs in ([^)]+)\))?\s*$", line)
    if m:
      impl = m.group(1)
      rows.append(("FAIL", impl, m.group(2) or impl, "opentitan", "LEC"))

rows.sort(key=lambda r: (r[1], r[0], r[2]))
with out_path.open("w") as f:
  for row in rows:
    f.write("\t".join(row) + "\n")
PY
  fi
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
  avip_case_results="$OUT_DIR/avip-results.txt"
  : > "$avip_case_results"
  for avip in $AVIP_GLOB; do
    if [[ -d "$avip" ]]; then
      avip_name="$(basename "$avip")"
      avip_lane_id="avip/${avip_name}/compile"
      if ! lane_enabled "$avip_lane_id"; then
        continue
      fi
      run_suite "avip-${avip_name}" \
        env OUT="$OUT_DIR/${avip_name}-circt-verilog.log" \
        CIRCT_VERILOG="$CIRCT_VERILOG_BIN_AVIP" \
        utils/run_avip_circt_verilog.sh "$avip" || true
      if [[ -f "$OUT_DIR/avip-${avip_name}.exit" ]]; then
        avip_ec=$(cat "$OUT_DIR/avip-${avip_name}.exit")
        record_simple_result "avip/${avip_name}" "compile" "$avip_ec"
        avip_status="FAIL"
        if [[ "$avip_ec" == "0" ]]; then
          avip_status="PASS"
        fi
        printf "%s\t%s\t%s\t%s\t%s\n" \
          "$avip_status" "$avip_name" "$avip" "avip/${avip_name}" "compile" >> "$avip_case_results"
      fi
    fi
  done
  sort -o "$avip_case_results" "$avip_case_results"
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

if [[ -n "$EXPECTED_FAILURES_FILE" || \
      "$FAIL_ON_UNEXPECTED_FAILURES" == "1" || \
      "$FAIL_ON_UNUSED_EXPECTED_FAILURES" == "1" || \
      -n "$PRUNE_EXPECTED_FAILURES_FILE" ]]; then
  OUT_DIR="$OUT_DIR" \
  JSON_SUMMARY_FILE="$JSON_SUMMARY_FILE" \
  EXPECTED_FAILURES_FILE="$EXPECTED_FAILURES_FILE" \
  FAIL_ON_UNEXPECTED_FAILURES="$FAIL_ON_UNEXPECTED_FAILURES" \
  FAIL_ON_UNUSED_EXPECTED_FAILURES="$FAIL_ON_UNUSED_EXPECTED_FAILURES" \
  EXPECTATIONS_DRY_RUN="$EXPECTATIONS_DRY_RUN" \
  EXPECTATIONS_DRY_RUN_RUN_ID="$RUN_ID" \
  EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS="$EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS" \
  EXPECTATIONS_DRY_RUN_REPORT_JSONL="$EXPECTATIONS_DRY_RUN_REPORT_JSONL" \
  PRUNE_EXPECTED_FAILURES_FILE="$PRUNE_EXPECTED_FAILURES_FILE" \
  PRUNE_EXPECTED_FAILURES_DROP_UNUSED="$PRUNE_EXPECTED_FAILURES_DROP_UNUSED" \
  python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
summary_path = out_dir / "summary.tsv"
json_summary_path = Path(os.environ["JSON_SUMMARY_FILE"])
expected_file_raw = os.environ.get("EXPECTED_FAILURES_FILE", "")
fail_on_unexpected = os.environ.get("FAIL_ON_UNEXPECTED_FAILURES", "0") == "1"
fail_on_unused = os.environ.get("FAIL_ON_UNUSED_EXPECTED_FAILURES", "0") == "1"
expectations_dry_run = os.environ.get("EXPECTATIONS_DRY_RUN", "0") == "1"
dry_run_run_id = os.environ.get("EXPECTATIONS_DRY_RUN_RUN_ID", "")
max_sample_rows = int(
    os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS", "5")
)
dry_run_report_jsonl_raw = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_JSONL", "")
prune_file_raw = os.environ.get("PRUNE_EXPECTED_FAILURES_FILE", "")
prune_drop_unused = (
    os.environ.get("PRUNE_EXPECTED_FAILURES_DROP_UNUSED", "0") == "1"
)
expected_path = Path(expected_file_raw) if expected_file_raw else None
prune_path = Path(prune_file_raw) if prune_file_raw else None
dry_run_report_jsonl_path = Path(dry_run_report_jsonl_raw) if dry_run_report_jsonl_raw else None
expected_summary_path = out_dir / "expected-failures-summary.tsv"

def emit_dry_run_report(payload):
  if dry_run_report_jsonl_path is None:
    return
  payload = dict(payload)
  payload.setdefault("run_id", dry_run_run_id)
  dry_run_report_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
  with dry_run_report_jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")

def sample_rows(rows):
  if max_sample_rows <= 0:
    return []
  return rows[:max_sample_rows]

required_cols = {"suite", "mode", "expected_fail", "expected_error"}
expected = {}
if expected_path is not None:
  with expected_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    fieldnames = set(reader.fieldnames or [])
    if not required_cols.issubset(fieldnames):
      missing = sorted(required_cols - fieldnames)
      raise SystemExit(
          f"invalid expected-failures file: missing required columns: {', '.join(missing)}"
      )
    for row in reader:
      suite = row.get("suite", "").strip()
      mode = row.get("mode", "").strip()
      if not suite or not mode:
        raise SystemExit("invalid expected-failures row: suite/mode must be non-empty")
      key = (suite, mode)
      if key in expected:
        raise SystemExit(
            f"invalid expected-failures file: duplicate row for suite={suite} mode={mode}"
        )
      try:
        expected_fail = int(row.get("expected_fail", "0"))
        expected_error = int(row.get("expected_error", "0"))
      except ValueError:
        raise SystemExit(
            f"invalid expected-failures row for suite={suite} mode={mode}: "
            "expected_fail/expected_error must be integers"
        )
      if expected_fail < 0 or expected_error < 0:
        raise SystemExit(
            f"invalid expected-failures row for suite={suite} mode={mode}: "
            "expected_fail/expected_error must be non-negative"
        )
      expected[key] = {
          "expected_fail": expected_fail,
          "expected_error": expected_error,
          "notes": row.get("notes", ""),
      }

rows = []
seen = set()
totals = {
    "actual_fail": 0,
    "actual_error": 0,
    "expected_fail": 0,
    "expected_error": 0,
    "unexpected_fail": 0,
    "unexpected_error": 0,
}

with summary_path.open() as f:
  reader = csv.DictReader(f, delimiter='\t')
  for row in reader:
    suite = row["suite"]
    mode = row["mode"]
    key = (suite, mode)
    seen.add(key)
    actual_fail = int(row["fail"])
    actual_error = int(row["error"])
    exp = expected.get(key, {"expected_fail": 0, "expected_error": 0, "notes": ""})
    expected_fail = exp["expected_fail"]
    expected_error = exp["expected_error"]
    unexpected_fail = max(actual_fail - expected_fail, 0)
    unexpected_error = max(actual_error - expected_error, 0)
    totals["actual_fail"] += actual_fail
    totals["actual_error"] += actual_error
    totals["expected_fail"] += expected_fail
    totals["expected_error"] += expected_error
    totals["unexpected_fail"] += unexpected_fail
    totals["unexpected_error"] += unexpected_error
    rows.append(
        {
            "suite": suite,
            "mode": mode,
            "fail": actual_fail,
            "error": actual_error,
            "expected_fail": expected_fail,
            "expected_error": expected_error,
            "unexpected_fail": unexpected_fail,
            "unexpected_error": unexpected_error,
            "within_budget": "yes" if unexpected_fail == 0 and unexpected_error == 0 else "no",
            "notes": exp["notes"],
        }
    )

unused_expectations = []
for suite, mode in sorted(expected.keys() - seen):
  unused_expectations.append({"suite": suite, "mode": mode})

with expected_summary_path.open("w", newline="") as f:
  writer = csv.DictWriter(
      f,
      fieldnames=[
          "suite",
          "mode",
          "fail",
          "error",
          "expected_fail",
          "expected_error",
          "unexpected_fail",
          "unexpected_error",
          "within_budget",
          "notes",
      ],
      delimiter='\t',
  )
  writer.writeheader()
  for row in rows:
    writer.writerow(row)

try:
  payload = json.loads(json_summary_path.read_text())
except Exception:
  payload = {}

payload["expected_failures"] = {
    "file": str(expected_path) if expected_path is not None else "",
    "rows": rows,
    "totals": totals,
    "unused_expectations": unused_expectations,
}
json_summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

print(f"expected-failures summary: {expected_summary_path}")
print(
    "expected-failures totals: "
    f"actual_fail={totals['actual_fail']} actual_error={totals['actual_error']} "
    f"expected_fail={totals['expected_fail']} expected_error={totals['expected_error']} "
    f"unexpected_fail={totals['unexpected_fail']} unexpected_error={totals['unexpected_error']}"
)
if unused_expectations:
  print("expected-failures unused entries:")
  for item in unused_expectations:
    print(f"  {item['suite']} {item['mode']}")

if prune_path is not None:
  if expected_path is None:
    raise SystemExit("--prune-expected-failures-file requires expected failures to be loaded")
  pruned_rows = []
  dropped_rows = []
  dropped_unused = 0
  for (suite, mode), exp in expected.items():
    if prune_drop_unused and (suite, mode) not in seen:
      dropped_unused += 1
      dropped_rows.append(
          {
              "suite": suite,
              "mode": mode,
              "expected_fail": exp["expected_fail"],
              "expected_error": exp["expected_error"],
              "notes": exp.get("notes", ""),
              "drop_reason": "unused",
          }
      )
      continue
    pruned_rows.append(
        {
            "suite": suite,
            "mode": mode,
            "expected_fail": exp["expected_fail"],
            "expected_error": exp["expected_error"],
            "notes": exp.get("notes", ""),
        }
    )
  prune_path.parent.mkdir(parents=True, exist_ok=True)
  if expectations_dry_run:
    print(f"dry-run: would prune expected-failures file: {prune_path}")
    emit_dry_run_report(
        {
            "operation": "prune_expected_failures",
            "target_file": str(prune_path),
            "kept_rows": len(pruned_rows),
            "dropped_unused": dropped_unused,
            "kept_rows_sample": sample_rows(pruned_rows),
            "dropped_rows_sample": sample_rows(dropped_rows),
        }
    )
  else:
    with prune_path.open("w", newline="") as f:
      writer = csv.DictWriter(
          f,
          fieldnames=["suite", "mode", "expected_fail", "expected_error", "notes"],
          delimiter="\t",
      )
      writer.writeheader()
      for row in pruned_rows:
        writer.writerow(row)
    print(f"pruned expected-failures file: {prune_path}")
  print(
      "pruned expected-failures rows: "
      f"kept={len(pruned_rows)} dropped_unused={dropped_unused}"
  )

if fail_on_unexpected and (
    totals["unexpected_fail"] > 0 or totals["unexpected_error"] > 0
):
  print("unexpected failure budget overruns:")
  for row in rows:
    if row["unexpected_fail"] > 0 or row["unexpected_error"] > 0:
      print(
          f"  {row['suite']} {row['mode']}: "
          f"fail {row['fail']} (expected {row['expected_fail']}), "
          f"error {row['error']} (expected {row['expected_error']})"
      )
  raise SystemExit(1)

if fail_on_unused and unused_expectations:
  print("unused expected-failures entries:")
  for item in unused_expectations:
    print(f"  {item['suite']} {item['mode']}")
  raise SystemExit(1)
PY
fi

if [[ -n "$REFRESH_EXPECTED_FAILURES_FILE" ]]; then
  OUT_DIR="$OUT_DIR" \
  SUMMARY_FILE="$OUT_DIR/summary.tsv" \
  EXPECTATIONS_DRY_RUN="$EXPECTATIONS_DRY_RUN" \
  EXPECTATIONS_DRY_RUN_RUN_ID="$RUN_ID" \
  EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS="$EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS" \
  EXPECTATIONS_DRY_RUN_REPORT_JSONL="$EXPECTATIONS_DRY_RUN_REPORT_JSONL" \
  REFRESH_EXPECTED_FAILURES_FILE="$REFRESH_EXPECTED_FAILURES_FILE" \
  REFRESH_EXPECTED_FAILURES_INCLUDE_SUITE_REGEX="$REFRESH_EXPECTED_FAILURES_INCLUDE_SUITE_REGEX" \
  REFRESH_EXPECTED_FAILURES_INCLUDE_MODE_REGEX="$REFRESH_EXPECTED_FAILURES_INCLUDE_MODE_REGEX" \
  python3 - <<'PY'
import csv
import json
import os
import re
from pathlib import Path

summary_path = Path(os.environ["SUMMARY_FILE"])
out_path = Path(os.environ["REFRESH_EXPECTED_FAILURES_FILE"])
expectations_dry_run = os.environ.get("EXPECTATIONS_DRY_RUN", "0") == "1"
dry_run_run_id = os.environ.get("EXPECTATIONS_DRY_RUN_RUN_ID", "")
max_sample_rows = int(
    os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS", "5")
)
dry_run_report_jsonl_raw = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_JSONL", "")
dry_run_report_jsonl_path = Path(dry_run_report_jsonl_raw) if dry_run_report_jsonl_raw else None
suite_filter_raw = os.environ.get("REFRESH_EXPECTED_FAILURES_INCLUDE_SUITE_REGEX", "")
mode_filter_raw = os.environ.get("REFRESH_EXPECTED_FAILURES_INCLUDE_MODE_REGEX", "")

def emit_dry_run_report(payload):
  if dry_run_report_jsonl_path is None:
    return
  payload = dict(payload)
  payload.setdefault("run_id", dry_run_run_id)
  dry_run_report_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
  with dry_run_report_jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")

def sample_rows(rows):
  if max_sample_rows <= 0:
    return []
  return rows[:max_sample_rows]

def compile_optional_regex(raw: str, field: str):
  if not raw:
    return None
  try:
    return re.compile(raw)
  except re.error as ex:
    raise SystemExit(f"invalid {field}: {ex}")

suite_filter = compile_optional_regex(
    suite_filter_raw, "--refresh-expected-failures-include-suite-regex"
)
mode_filter = compile_optional_regex(
    mode_filter_raw, "--refresh-expected-failures-include-mode-regex"
)

existing_notes = {}
if out_path.exists():
  with out_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
      suite = (row.get("suite") or "").strip()
      mode = (row.get("mode") or "").strip()
      if not suite or not mode:
        continue
      existing_notes[(suite, mode)] = row.get("notes", "")

rows = []
with summary_path.open() as f:
  reader = csv.DictReader(f, delimiter="\t")
  for row in reader:
    suite = row["suite"]
    mode = row["mode"]
    if suite_filter is not None and suite_filter.search(suite) is None:
      continue
    if mode_filter is not None and mode_filter.search(mode) is None:
      continue
    rows.append(
        {
            "suite": suite,
            "mode": mode,
            "expected_fail": int(row.get("fail", "0") or 0),
            "expected_error": int(row.get("error", "0") or 0),
            "notes": existing_notes.get((suite, mode), ""),
        }
    )

rows.sort(key=lambda r: (r["suite"], r["mode"]))
out_path.parent.mkdir(parents=True, exist_ok=True)
if expectations_dry_run:
  print(f"dry-run: would refresh expected-failures file: {out_path}")
  emit_dry_run_report(
      {
          "operation": "refresh_expected_failures",
          "target_file": str(out_path),
          "output_rows": len(rows),
          "output_rows_sample": sample_rows(rows),
          "suite_filter": suite_filter_raw,
          "mode_filter": mode_filter_raw,
      }
  )
else:
  with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["suite", "mode", "expected_fail", "expected_error", "notes"],
        delimiter="\t",
    )
    writer.writeheader()
    for row in rows:
      writer.writerow(row)
  print(f"refreshed expected-failures file: {out_path}")
print(f"refreshed expected-failures rows: {len(rows)}")
PY
fi

if [[ -n "$EXPECTED_FAILURE_CASES_FILE" || \
      "$FAIL_ON_UNEXPECTED_FAILURE_CASES" == "1" || \
      "$FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES" == "1" || \
      "$FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES" == "1" || \
      -n "$PRUNE_EXPECTED_FAILURE_CASES_FILE" ]]; then
  OUT_DIR="$OUT_DIR" \
  JSON_SUMMARY_FILE="$JSON_SUMMARY_FILE" \
  EXPECTED_FAILURE_CASES_FILE="$EXPECTED_FAILURE_CASES_FILE" \
  FAIL_ON_UNEXPECTED_FAILURE_CASES="$FAIL_ON_UNEXPECTED_FAILURE_CASES" \
  FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES="$FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES" \
  FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES="$FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES" \
  EXPECTATIONS_DRY_RUN="$EXPECTATIONS_DRY_RUN" \
  EXPECTATIONS_DRY_RUN_RUN_ID="$RUN_ID" \
  EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS="$EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS" \
  EXPECTATIONS_DRY_RUN_REPORT_JSONL="$EXPECTATIONS_DRY_RUN_REPORT_JSONL" \
  PRUNE_EXPECTED_FAILURE_CASES_FILE="$PRUNE_EXPECTED_FAILURE_CASES_FILE" \
  PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED="$PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED" \
  PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED="$PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED" \
  python3 - <<'PY'
import csv
import datetime as dt
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
json_summary_path = Path(os.environ["JSON_SUMMARY_FILE"])
expected_file_raw = os.environ.get("EXPECTED_FAILURE_CASES_FILE", "")
expected_path = Path(expected_file_raw) if expected_file_raw else None
fail_on_unexpected = os.environ.get("FAIL_ON_UNEXPECTED_FAILURE_CASES", "0") == "1"
fail_on_expired = os.environ.get("FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES", "0") == "1"
fail_on_unmatched = os.environ.get("FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES", "0") == "1"
expectations_dry_run = os.environ.get("EXPECTATIONS_DRY_RUN", "0") == "1"
dry_run_run_id = os.environ.get("EXPECTATIONS_DRY_RUN_RUN_ID", "")
max_sample_rows = int(
    os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS", "5")
)
dry_run_report_jsonl_raw = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_JSONL", "")
prune_file_raw = os.environ.get("PRUNE_EXPECTED_FAILURE_CASES_FILE", "")
prune_path = Path(prune_file_raw) if prune_file_raw else None
dry_run_report_jsonl_path = Path(dry_run_report_jsonl_raw) if dry_run_report_jsonl_raw else None
prune_drop_unmatched = (
    os.environ.get("PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED", "0") == "1"
)
prune_drop_expired = (
    os.environ.get("PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED", "0") == "1"
)
today = dt.date.today()

case_summary_path = out_dir / "expected-failure-cases-summary.tsv"
unexpected_path = out_dir / "unexpected-failure-cases.tsv"

def emit_dry_run_report(payload):
  if dry_run_report_jsonl_path is None:
    return
  payload = dict(payload)
  payload.setdefault("run_id", dry_run_run_id)
  dry_run_report_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
  with dry_run_report_jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")

def sample_rows(rows):
  if max_sample_rows <= 0:
    return []
  return rows[:max_sample_rows]

fail_like_statuses = {"FAIL", "ERROR", "XFAIL", "XPASS", "EFAIL"}
result_sources = [
    ("sv-tests", "BMC", out_dir / "sv-tests-bmc-results.txt"),
    ("sv-tests", "LEC", out_dir / "sv-tests-lec-results.txt"),
    ("verilator-verification", "BMC", out_dir / "verilator-bmc-results.txt"),
    ("verilator-verification", "LEC", out_dir / "verilator-lec-results.txt"),
    ("yosys/tests/sva", "LEC", out_dir / "yosys-lec-results.txt"),
    ("opentitan", "LEC", out_dir / "opentitan-lec-results.txt"),
    ("", "", out_dir / "avip-results.txt"),
]
detailed_source_pairs = {
    (suite, mode) for suite, mode, _ in result_sources if suite and mode
}
detailed_pairs_observed = set()

observed = []
for default_suite, default_mode, path in result_sources:
  if not path.exists():
    continue
  with path.open() as f:
    for line in f:
      line = line.rstrip("\n")
      if not line:
        continue
      parts = line.split("\t")
      status = parts[0].strip().upper() if parts else ""
      if status not in fail_like_statuses:
        continue
      suite = (
          parts[3].strip() if len(parts) > 3 and parts[3].strip() else default_suite
      )
      mode = (
          parts[4].strip() if len(parts) > 4 and parts[4].strip() else default_mode
      )
      if not suite or not mode:
        continue
      base = parts[1].strip() if len(parts) > 1 else ""
      file_path = parts[2].strip() if len(parts) > 2 else ""
      detailed_pairs_observed.add((suite, mode))
      observed.append(
          {
              "suite": suite,
              "mode": mode,
              "status": status,
              "base": base,
              "path": file_path,
          }
      )

summary_path = out_dir / "summary.tsv"
if summary_path.exists():
  with summary_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
      suite = row.get("suite", "")
      mode = row.get("mode", "")
      if not suite or not mode:
        continue
      if (suite, mode) in detailed_source_pairs or (suite, mode) in detailed_pairs_observed:
        continue
      summary = row.get("summary", "")
      try:
        fail_count = int(row.get("fail", "0"))
      except Exception:
        fail_count = 0
      try:
        error_count = int(row.get("error", "0"))
      except Exception:
        error_count = 0
      try:
        xfail_count = int(row.get("xfail", "0"))
      except Exception:
        xfail_count = 0
      try:
        xpass_count = int(row.get("xpass", "0"))
      except Exception:
        xpass_count = 0
      if fail_count > 0:
        observed.append(
            {
                "suite": suite,
                "mode": mode,
                "status": "FAIL",
                "base": "__aggregate__",
                "path": summary,
            }
        )
      if error_count > 0:
        observed.append(
            {
                "suite": suite,
                "mode": mode,
                "status": "ERROR",
                "base": "__aggregate__",
                "path": summary,
            }
        )
      if xfail_count > 0:
        observed.append(
            {
                "suite": suite,
                "mode": mode,
                "status": "XFAIL",
                "base": "__aggregate__",
                "path": summary,
            }
        )
      if xpass_count > 0:
        observed.append(
            {
                "suite": suite,
                "mode": mode,
                "status": "XPASS",
                "base": "__aggregate__",
                "path": summary,
            }
        )

expected_rows = []
if expected_path is not None:
  with expected_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    required_cols = {"suite", "mode", "id"}
    fieldnames = set(reader.fieldnames or [])
    if not required_cols.issubset(fieldnames):
      missing = sorted(required_cols - fieldnames)
      raise SystemExit(
          "invalid expected-failure-cases file: "
          f"missing required columns: {', '.join(missing)}"
      )
    seen = set()
    for idx, row in enumerate(reader):
      suite = row.get("suite", "").strip()
      mode = row.get("mode", "").strip()
      case_id = row.get("id", "").strip()
      if not suite or not mode or not case_id:
        raise SystemExit(
            "invalid expected-failure-cases row: "
            f"suite/mode/id must be non-empty at data row {idx + 1}"
        )
      id_kind = row.get("id_kind", "base").strip().lower() or "base"
      if id_kind not in {"base", "path", "aggregate"}:
        raise SystemExit(
            "invalid expected-failure-cases row for "
            f"suite={suite} mode={mode} id={case_id}: "
            "id_kind must be one of base,path,aggregate"
        )
      status = row.get("status", "ANY").strip().upper() or "ANY"
      if status != "ANY" and status not in fail_like_statuses:
        raise SystemExit(
            "invalid expected-failure-cases row for "
            f"suite={suite} mode={mode} id={case_id}: "
            f"unsupported status '{status}'"
        )
      expires_on = row.get("expires_on", "").strip()
      expires_date = None
      if expires_on:
        try:
          expires_date = dt.date.fromisoformat(expires_on)
        except Exception:
          raise SystemExit(
              "invalid expected-failure-cases row for "
              f"suite={suite} mode={mode} id={case_id}: "
              f"invalid expires_on '{expires_on}' (expected YYYY-MM-DD)"
          )
      key = (suite, mode, id_kind, case_id, status)
      if key in seen:
        raise SystemExit(
            "invalid expected-failure-cases file: "
            f"duplicate row for suite={suite} mode={mode} "
            f"id_kind={id_kind} id={case_id} status={status}"
        )
      seen.add(key)
      expected_rows.append(
          {
              "suite": suite,
              "mode": mode,
              "id_kind": id_kind,
              "id": case_id,
              "status": status,
              "expires_on": expires_on,
              "expires_date": expires_date,
              "reason": row.get("reason", ""),
          }
      )

matched_observed_idx = set()
expected_summary_rows = []
for row in expected_rows:
  matches = []
  for idx, obs in enumerate(observed):
    if obs["suite"] != row["suite"] or obs["mode"] != row["mode"]:
      continue
    if row["id_kind"] == "base":
      observed_id = obs["base"]
    elif row["id_kind"] == "path":
      observed_id = obs["path"]
    else:
      observed_id = "__aggregate__"
    if observed_id != row["id"]:
      continue
    if row["status"] != "ANY" and obs["status"] != row["status"]:
      continue
    matches.append(idx)
  for idx in matches:
    matched_observed_idx.add(idx)
  expired = "yes" if row["expires_date"] is not None and row["expires_date"] < today else "no"
  expected_summary_rows.append(
      {
          "suite": row["suite"],
          "mode": row["mode"],
          "id_kind": row["id_kind"],
          "id": row["id"],
          "status": row["status"],
          "expires_on": row["expires_on"],
          "matched_count": len(matches),
          "expired": expired,
          "reason": row["reason"],
      }
  )

unexpected_observed = []
for idx, obs in enumerate(observed):
  if idx in matched_observed_idx:
    continue
  unexpected_observed.append(obs)

with case_summary_path.open("w", newline="") as f:
  writer = csv.DictWriter(
      f,
      fieldnames=[
          "suite",
          "mode",
          "id_kind",
          "id",
          "status",
          "expires_on",
          "matched_count",
          "expired",
          "reason",
      ],
      delimiter="\t",
  )
  writer.writeheader()
  for row in expected_summary_rows:
    writer.writerow(row)

with unexpected_path.open("w", newline="") as f:
  writer = csv.DictWriter(
      f,
      fieldnames=["suite", "mode", "status", "base", "path"],
      delimiter="\t",
  )
  writer.writeheader()
  for row in unexpected_observed:
    writer.writerow(row)

expired_rows = [row for row in expected_summary_rows if row["expired"] == "yes"]
unmatched_rows = [row for row in expected_summary_rows if row["matched_count"] == 0]
totals = {
    "observed_fail_like": len(observed),
    "matched_expected": len(matched_observed_idx),
    "unexpected_observed": len(unexpected_observed),
    "expected_rows": len(expected_summary_rows),
    "unmatched_expected": len(unmatched_rows),
    "expired_expected": len(expired_rows),
}

try:
  payload = json.loads(json_summary_path.read_text())
except Exception:
  payload = {}

payload["expected_failure_cases"] = {
    "file": str(expected_path) if expected_path is not None else "",
    "rows": expected_summary_rows,
    "unexpected_observed": unexpected_observed,
    "totals": totals,
}
json_summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

print(f"expected-failure-cases summary: {case_summary_path}")
print(f"unexpected-failure-cases summary: {unexpected_path}")
print(
    "expected-failure-cases totals: "
    f"observed_fail_like={totals['observed_fail_like']} "
    f"matched_expected={totals['matched_expected']} "
    f"unexpected_observed={totals['unexpected_observed']} "
    f"expired_expected={totals['expired_expected']} "
    f"unmatched_expected={totals['unmatched_expected']}"
)

if prune_path is not None:
  if expected_path is None:
    raise SystemExit(
        "--prune-expected-failure-cases-file requires expected cases to be loaded"
    )
  pruned_rows = []
  dropped_rows = []
  dropped_unmatched = 0
  dropped_expired = 0
  for row, summary_row in zip(expected_rows, expected_summary_rows):
    is_unmatched = summary_row["matched_count"] == 0
    is_expired = summary_row["expired"] == "yes"
    drop = False
    drop_reasons = []
    if prune_drop_unmatched and is_unmatched:
      drop = True
      dropped_unmatched += 1
      drop_reasons.append("unmatched")
    if prune_drop_expired and is_expired:
      drop = True
      dropped_expired += 1
      drop_reasons.append("expired")
    if drop:
      dropped_rows.append(
          {
              "suite": row["suite"],
              "mode": row["mode"],
              "id": row["id"],
              "id_kind": row["id_kind"],
              "status": row["status"],
              "expires_on": row["expires_on"],
              "reason": row["reason"],
              "drop_reasons": ",".join(drop_reasons),
          }
      )
      continue
    pruned_rows.append(
        {
            "suite": row["suite"],
            "mode": row["mode"],
            "id": row["id"],
            "id_kind": row["id_kind"],
            "status": row["status"],
            "expires_on": row["expires_on"],
            "reason": row["reason"],
        }
    )

  prune_path.parent.mkdir(parents=True, exist_ok=True)
  if expectations_dry_run:
    print(f"dry-run: would prune expected-failure-cases file: {prune_path}")
    emit_dry_run_report(
        {
            "operation": "prune_expected_failure_cases",
            "target_file": str(prune_path),
            "kept_rows": len(pruned_rows),
            "dropped_unmatched": dropped_unmatched,
            "dropped_expired": dropped_expired,
            "kept_rows_sample": sample_rows(pruned_rows),
            "dropped_rows_sample": sample_rows(dropped_rows),
        }
    )
  else:
    with prune_path.open("w", newline="") as f:
      writer = csv.DictWriter(
          f,
          fieldnames=["suite", "mode", "id", "id_kind", "status", "expires_on", "reason"],
          delimiter="\t",
      )
      writer.writeheader()
      for row in pruned_rows:
        writer.writerow(row)
    print(f"pruned expected-failure-cases file: {prune_path}")
  print(
      "pruned expected-failure-cases rows: "
      f"kept={len(pruned_rows)} dropped_unmatched={dropped_unmatched} "
      f"dropped_expired={dropped_expired}"
  )

if fail_on_unexpected and unexpected_observed:
  print("unexpected observed failure cases:")
  for row in unexpected_observed:
    print(
        f"  {row['suite']} {row['mode']} {row['status']} "
        f"base={row['base']} path={row['path']}"
    )
  raise SystemExit(1)

if fail_on_expired and expired_rows:
  print("expired expected failure cases:")
  for row in expired_rows:
    print(
        f"  {row['suite']} {row['mode']} id_kind={row['id_kind']} "
        f"id={row['id']} expires_on={row['expires_on']} matched_count={row['matched_count']}"
      )
  raise SystemExit(1)

if fail_on_unmatched and unmatched_rows:
  print("unmatched expected failure cases:")
  for row in unmatched_rows:
    print(
        f"  {row['suite']} {row['mode']} id_kind={row['id_kind']} "
        f"id={row['id']} status={row['status']}"
    )
  raise SystemExit(1)
PY
fi

if [[ -n "$REFRESH_EXPECTED_FAILURE_CASES_FILE" ]]; then
  OUT_DIR="$OUT_DIR" \
  EXPECTATIONS_DRY_RUN="$EXPECTATIONS_DRY_RUN" \
  EXPECTATIONS_DRY_RUN_RUN_ID="$RUN_ID" \
  EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS="$EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS" \
  EXPECTATIONS_DRY_RUN_REPORT_JSONL="$EXPECTATIONS_DRY_RUN_REPORT_JSONL" \
  REFRESH_EXPECTED_FAILURE_CASES_FILE="$REFRESH_EXPECTED_FAILURE_CASES_FILE" \
  REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON="$REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON" \
  REFRESH_EXPECTED_FAILURE_CASES_COLLAPSE_STATUS_ANY="$REFRESH_EXPECTED_FAILURE_CASES_COLLAPSE_STATUS_ANY" \
  REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_SUITE_REGEX="$REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_SUITE_REGEX" \
  REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_MODE_REGEX="$REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_MODE_REGEX" \
  REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_STATUS_REGEX="$REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_STATUS_REGEX" \
  REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_ID_REGEX="$REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_ID_REGEX" \
  python3 - <<'PY'
import csv
import datetime as dt
import json
import os
import re
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
out_path = Path(os.environ["REFRESH_EXPECTED_FAILURE_CASES_FILE"])
expectations_dry_run = os.environ.get("EXPECTATIONS_DRY_RUN", "0") == "1"
dry_run_run_id = os.environ.get("EXPECTATIONS_DRY_RUN_RUN_ID", "")
max_sample_rows = int(
    os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS", "5")
)
dry_run_report_jsonl_raw = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_JSONL", "")
dry_run_report_jsonl_path = Path(dry_run_report_jsonl_raw) if dry_run_report_jsonl_raw else None
default_expires = os.environ.get("REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON", "").strip()
collapse_status_any = (
    os.environ.get("REFRESH_EXPECTED_FAILURE_CASES_COLLAPSE_STATUS_ANY", "0") == "1"
)
suite_filter_raw = os.environ.get(
    "REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_SUITE_REGEX", ""
)
mode_filter_raw = os.environ.get(
    "REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_MODE_REGEX", ""
)
status_filter_raw = os.environ.get(
    "REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_STATUS_REGEX", ""
)
id_filter_raw = os.environ.get("REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_ID_REGEX", "")

def emit_dry_run_report(payload):
  if dry_run_report_jsonl_path is None:
    return
  payload = dict(payload)
  payload.setdefault("run_id", dry_run_run_id)
  dry_run_report_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
  with dry_run_report_jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")

def sample_rows(rows):
  if max_sample_rows <= 0:
    return []
  return rows[:max_sample_rows]

def compile_optional_regex(raw: str, field: str):
  if not raw:
    return None
  try:
    return re.compile(raw)
  except re.error as ex:
    raise SystemExit(f"invalid {field}: {ex}")

suite_filter = compile_optional_regex(
    suite_filter_raw, "--refresh-expected-failure-cases-include-suite-regex"
)
mode_filter = compile_optional_regex(
    mode_filter_raw, "--refresh-expected-failure-cases-include-mode-regex"
)
status_filter = compile_optional_regex(
    status_filter_raw, "--refresh-expected-failure-cases-include-status-regex"
)
id_filter = compile_optional_regex(
    id_filter_raw, "--refresh-expected-failure-cases-include-id-regex"
)

if default_expires:
  try:
    dt.date.fromisoformat(default_expires)
  except Exception:
    raise SystemExit(
        "invalid --refresh-expected-failure-cases-default-expires-on: "
        f"{default_expires}"
    )

existing_meta = {}
if out_path.exists():
  with out_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
      suite = (row.get("suite") or "").strip()
      mode = (row.get("mode") or "").strip()
      case_id = (row.get("id") or "").strip()
      id_kind = (row.get("id_kind") or "base").strip().lower() or "base"
      status = (row.get("status") or "ANY").strip().upper() or "ANY"
      if not suite or not mode or not case_id:
        continue
      key = (suite, mode, id_kind, case_id, status)
      existing_meta[key] = {
          "expires_on": (row.get("expires_on") or "").strip(),
          "reason": row.get("reason", ""),
      }

fail_like_statuses = {"FAIL", "ERROR", "XFAIL", "XPASS", "EFAIL"}
result_sources = [
    ("sv-tests", "BMC", out_dir / "sv-tests-bmc-results.txt"),
    ("sv-tests", "LEC", out_dir / "sv-tests-lec-results.txt"),
    ("verilator-verification", "BMC", out_dir / "verilator-bmc-results.txt"),
    ("verilator-verification", "LEC", out_dir / "verilator-lec-results.txt"),
    ("yosys/tests/sva", "LEC", out_dir / "yosys-lec-results.txt"),
    ("opentitan", "LEC", out_dir / "opentitan-lec-results.txt"),
    ("", "", out_dir / "avip-results.txt"),
]
detailed_pairs_observed = set()

observed = []
for default_suite, default_mode, path in result_sources:
  if not path.exists():
    continue
  with path.open() as f:
    for line in f:
      line = line.rstrip("\n")
      if not line:
        continue
      parts = line.split("\t")
      status = parts[0].strip().upper() if parts else ""
      if status not in fail_like_statuses:
        continue
      suite = (
          parts[3].strip() if len(parts) > 3 and parts[3].strip() else default_suite
      )
      mode = (
          parts[4].strip() if len(parts) > 4 and parts[4].strip() else default_mode
      )
      if not suite or not mode:
        continue
      base = parts[1].strip() if len(parts) > 1 else ""
      file_path = parts[2].strip() if len(parts) > 2 else ""
      detailed_pairs_observed.add((suite, mode))
      observed.append(
          {
              "suite": suite,
              "mode": mode,
              "status": status,
              "base": base,
              "path": file_path,
          }
      )

summary_path = out_dir / "summary.tsv"
if summary_path.exists():
  with summary_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
      suite = row.get("suite", "")
      mode = row.get("mode", "")
      if not suite or not mode:
        continue
      if (suite, mode) in detailed_pairs_observed:
        continue
      summary = row.get("summary", "")
      for summary_key, status in (
          ("fail", "FAIL"),
          ("error", "ERROR"),
          ("xfail", "XFAIL"),
          ("xpass", "XPASS"),
      ):
        try:
          count = int(row.get(summary_key, "0"))
        except Exception:
          count = 0
        if count <= 0:
          continue
        observed.append(
            {
                "suite": suite,
                "mode": mode,
                "status": status,
                "base": "__aggregate__",
                "path": summary,
            }
        )

refreshed = []
seen = set()

def derive_case_id(obs_row):
  if obs_row["base"] == "__aggregate__":
    return ("aggregate", "__aggregate__")
  if obs_row["base"]:
    return ("base", obs_row["base"])
  return ("path", obs_row["path"])

def include_obs(obs_row, id_kind: str, case_id: str):
  if suite_filter is not None and suite_filter.search(obs_row["suite"]) is None:
    return False
  if mode_filter is not None and mode_filter.search(obs_row["mode"]) is None:
    return False
  if status_filter is not None and status_filter.search(obs_row["status"]) is None:
    return False
  if id_filter is not None and id_filter.search(case_id) is None:
    return False
  return True

if collapse_status_any:
  grouped = {}
  for obs in observed:
    id_kind, case_id = derive_case_id(obs)
    if not include_obs(obs, id_kind, case_id):
      continue
    base_key = (obs["suite"], obs["mode"], id_kind, case_id)
    grouped.setdefault(base_key, set()).add(obs["status"])
  for base_key, statuses in grouped.items():
    suite, mode, id_kind, case_id = base_key
    key = (suite, mode, id_kind, case_id, "ANY")
    if key in seen:
      continue
    seen.add(key)
    meta = existing_meta.get(key, {})
    if not meta:
      for status in sorted(statuses):
        exact_key = (suite, mode, id_kind, case_id, status)
        if exact_key in existing_meta:
          meta = existing_meta[exact_key]
          break
    refreshed.append(
        {
            "suite": suite,
            "mode": mode,
            "id": case_id,
            "id_kind": id_kind,
            "status": "ANY",
            "expires_on": meta.get("expires_on", "") or default_expires,
            "reason": meta.get("reason", ""),
        }
    )
else:
  for obs in observed:
    id_kind, case_id = derive_case_id(obs)
    if not include_obs(obs, id_kind, case_id):
      continue
    key = (obs["suite"], obs["mode"], id_kind, case_id, obs["status"])
    if key in seen:
      continue
    seen.add(key)
    meta = existing_meta.get(key, {})
    refreshed.append(
        {
            "suite": obs["suite"],
            "mode": obs["mode"],
            "id": case_id,
            "id_kind": id_kind,
            "status": obs["status"],
            "expires_on": meta.get("expires_on", "") or default_expires,
            "reason": meta.get("reason", ""),
        }
    )

refreshed.sort(key=lambda r: (r["suite"], r["mode"], r["id_kind"], r["id"], r["status"]))
out_path.parent.mkdir(parents=True, exist_ok=True)
if expectations_dry_run:
  print(f"dry-run: would refresh expected-failure-cases file: {out_path}")
  emit_dry_run_report(
      {
          "operation": "refresh_expected_failure_cases",
          "target_file": str(out_path),
          "output_rows": len(refreshed),
          "output_rows_sample": sample_rows(refreshed),
          "collapse_status_any": collapse_status_any,
          "suite_filter": suite_filter_raw,
          "mode_filter": mode_filter_raw,
          "status_filter": status_filter_raw,
          "id_filter": id_filter_raw,
      }
  )
else:
  with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["suite", "mode", "id", "id_kind", "status", "expires_on", "reason"],
        delimiter="\t",
    )
    writer.writeheader()
    for row in refreshed:
      writer.writerow(row)
  print(f"refreshed expected-failure-cases file: {out_path}")
print(f"refreshed expected-failure-cases rows: {len(refreshed)}")
PY
fi

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
  BASELINE_WINDOW_DAYS="$BASELINE_WINDOW_DAYS" \
  FAIL_ON_NEW_XPASS="$FAIL_ON_NEW_XPASS" \
  FAIL_ON_PASSRATE_REGRESSION="$FAIL_ON_PASSRATE_REGRESSION" \
  STRICT_GATE="$STRICT_GATE" python3 - <<'PY'
import csv
import datetime as dt
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
baseline_window_days = int(os.environ.get("BASELINE_WINDOW_DAYS", "0"))

gate_errors = []
for key, current_row in summary.items():
    suite, mode = key
    history_rows = history.get(key, [])
    if not history_rows:
        if strict_gate:
            gate_errors.append(f"{suite} {mode}: missing baseline row")
        continue
    history_rows.sort(key=lambda r: r.get("date", ""))
    if baseline_window_days > 0:
        parsed_dates = []
        for row in history_rows:
            try:
                parsed_dates.append(dt.date.fromisoformat(row.get("date", "")))
            except Exception:
                parsed_dates.append(None)
        valid_dates = [d for d in parsed_dates if d is not None]
        if valid_dates:
            latest_date = max(valid_dates)
            cutoff = latest_date - dt.timedelta(days=baseline_window_days)
            filtered_rows = []
            for row, row_date in zip(history_rows, parsed_dates):
                if row_date is None:
                    continue
                if cutoff <= row_date <= latest_date:
                    filtered_rows.append(row)
            history_rows = filtered_rows
    if strict_gate and len(history_rows) < baseline_window:
        gate_errors.append(
            f"{suite} {mode}: insufficient baseline history ({len(history_rows)} < {baseline_window})"
        )
        continue
    if not history_rows:
        if strict_gate:
            gate_errors.append(
                f"{suite} {mode}: no baseline rows remain after baseline-window-days={baseline_window_days} filtering"
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

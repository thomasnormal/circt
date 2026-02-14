#!/usr/bin/env bash
# Run mutation-flow smoke/regression checks against local MCY example designs.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: run_mutation_mcy_examples.sh [options]

Run circt-mut mutation coverage lanes on local MCY examples.

Options:
  --examples-root DIR      MCY examples root (default: ~/mcy/examples)
  --out-dir DIR            Output directory (default: ./mutation-mcy-examples-results)
  --example NAME           Example to run (repeatable; default: bitcnt,picorv32_primes)
                           Supported: bitcnt, picorv32_primes
  --circt-mut PATH         circt-mut binary or command (default: auto-detect)
  --yosys PATH             yosys binary (default: yosys)
  --generate-count N       Mutations to generate in non-smoke mode (default: 32)
  --mutation-limit N       Per-example mutation limit (default: 8)
  --smoke                  Run smoke mode without yosys:
                           - use stub mutations file
                           - use identity fake create-mutated script
  --keep-work              Keep per-example helper scripts under out-dir/.work
  -h, --help               Show this help

Outputs:
  <out-dir>/summary.tsv    Aggregated example status/coverage summary
  <out-dir>/<example>/     Per-example run artifacts
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CIRCT_MUT="${SCRIPT_DIR}/../build-test/bin/circt-mut"

EXAMPLES_ROOT="${HOME}/mcy/examples"
OUT_DIR="${PWD}/mutation-mcy-examples-results"
CIRCT_MUT=""
YOSYS_BIN="${YOSYS:-yosys}"
GENERATE_COUNT=32
MUTATION_LIMIT=8
SMOKE=0
KEEP_WORK=0
EXAMPLE_IDS=()

is_pos_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

resolve_tool() {
  local tool="$1"
  if [[ "$tool" == */* ]]; then
    if [[ -x "$tool" ]]; then
      printf '%s\n' "$tool"
      return 0
    fi
    return 1
  fi
  local resolved
  resolved="$(command -v "$tool" 2>/dev/null || true)"
  if [[ -n "$resolved" && -x "$resolved" ]]; then
    printf '%s\n' "$resolved"
    return 0
  fi
  return 1
}

append_summary_row() {
  local summary_file="$1"
  local example_id="$2"
  local status="$3"
  local exit_code="$4"
  local detected="$5"
  local relevant="$6"
  local coverage="$7"
  local errors="$8"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$example_id" "$status" "$exit_code" "$detected" "$relevant" "$coverage" "$errors" \
    >> "$summary_file"
}

metric_value_or_zero() {
  local file="$1"
  local key="$2"
  local value
  value="$(awk -F '\t' -v k="$key" '$1==k { print $2; found=1; exit } END { if (!found) print 0 }' "$file" 2>/dev/null || true)"
  if [[ -z "$value" ]]; then
    printf '0\n'
  else
    printf '%s\n' "$value"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --examples-root)
      EXAMPLES_ROOT="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --example)
      EXAMPLE_IDS+=("$2")
      shift 2
      ;;
    --circt-mut)
      CIRCT_MUT="$2"
      shift 2
      ;;
    --yosys)
      YOSYS_BIN="$2"
      shift 2
      ;;
    --generate-count)
      GENERATE_COUNT="$2"
      shift 2
      ;;
    --mutation-limit)
      MUTATION_LIMIT="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --keep-work)
      KEEP_WORK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! is_pos_int "$GENERATE_COUNT"; then
  echo "--generate-count must be a positive integer: $GENERATE_COUNT" >&2
  exit 1
fi
if ! is_pos_int "$MUTATION_LIMIT"; then
  echo "--mutation-limit must be a positive integer: $MUTATION_LIMIT" >&2
  exit 1
fi

if [[ ${#EXAMPLE_IDS[@]} -eq 0 ]]; then
  EXAMPLE_IDS=("bitcnt" "picorv32_primes")
fi

if [[ -z "$CIRCT_MUT" ]]; then
  if [[ -x "$DEFAULT_CIRCT_MUT" ]]; then
    CIRCT_MUT="$DEFAULT_CIRCT_MUT"
  else
    CIRCT_MUT="circt-mut"
  fi
fi

if ! CIRCT_MUT_RESOLVED="$(resolve_tool "$CIRCT_MUT")"; then
  echo "circt-mut not found or not executable: $CIRCT_MUT" >&2
  exit 1
fi

if [[ "$SMOKE" -ne 1 ]]; then
  if ! YOSYS_RESOLVED="$(resolve_tool "$YOSYS_BIN")"; then
    echo "yosys not found or not executable: $YOSYS_BIN (use --smoke or --yosys PATH)" >&2
    exit 1
  fi
else
  YOSYS_RESOLVED=""
fi

mkdir -p "$OUT_DIR"
WORK_ROOT="${OUT_DIR}/.work"
mkdir -p "$WORK_ROOT"
if [[ "$KEEP_WORK" -ne 1 ]]; then
  trap 'rm -rf "$WORK_ROOT"' EXIT
fi

SUMMARY_FILE="${OUT_DIR}/summary.tsv"
printf 'example\tstatus\texit_code\tdetected\trelevant\tcoverage_percent\terrors\n' > "$SUMMARY_FILE"

overall_rc=0

for example_id in "${EXAMPLE_IDS[@]}"; do
  case "$example_id" in
    bitcnt)
      design="${EXAMPLES_ROOT}/bitcnt/bitcnt.v"
      top="bitcnt"
      ;;
    picorv32_primes)
      design="${EXAMPLES_ROOT}/picorv32_primes/picorv32.v"
      top="picorv32"
      ;;
    *)
      echo "Unsupported --example value: $example_id" >&2
      exit 1
      ;;
  esac

  if [[ ! -f "$design" ]]; then
    echo "Missing example design for ${example_id}: $design" >&2
    exit 1
  fi

  example_out_dir="${OUT_DIR}/${example_id}"
  helper_dir="${WORK_ROOT}/${example_id}"
  mkdir -p "$example_out_dir" "$helper_dir"

  fake_test_script="${helper_dir}/fake_test.sh"
  cat > "$fake_test_script" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
echo "SURVIVED" > result.txt
EOS
  chmod +x "$fake_test_script"

  tests_manifest="${helper_dir}/tests.tsv"
  printf 'smoke\tbash %s\tresult.txt\t^DETECTED$\t^SURVIVED$\n' "$fake_test_script" > "$tests_manifest"

  cmd=(
    "$CIRCT_MUT_RESOLVED" cover
    --design "$design"
    --tests-manifest "$tests_manifest"
    --work-dir "$example_out_dir"
    --skip-baseline
    --jobs 1
    --mutation-limit "$MUTATION_LIMIT"
  )

  if [[ "$SMOKE" -eq 1 ]]; then
    mutations_file="${helper_dir}/mutations.txt"
    printf '1 M_SMOKE_A\n2 M_SMOKE_B\n3 M_SMOKE_C\n' > "$mutations_file"
    fake_create_mutated="${helper_dir}/fake_create_mutated.sh"
    cat > "$fake_create_mutated" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
out=""
design=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      out="$2"
      shift 2
      ;;
    -d|--design)
      design="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "$out" || -z "$design" ]]; then
  echo "fake_create_mutated.sh missing -o/--output or -d/--design" >&2
  exit 2
fi
cp "$design" "$out"
EOS
    chmod +x "$fake_create_mutated"
    cmd+=(
      --mutations-file "$mutations_file"
      --create-mutated-script "$fake_create_mutated"
    )
  else
    cmd+=(
      --generate-mutations "$GENERATE_COUNT"
      --mutations-top "$top"
      --mutations-yosys "$YOSYS_RESOLVED"
    )
  fi

  run_log="${example_out_dir}/run.log"
  set +e
  "${cmd[@]}" >"$run_log" 2>&1
  rc=$?
  set -e

  metrics_file="${example_out_dir}/metrics.tsv"
  detected="0"
  relevant="0"
  errors="0"
  coverage="-"
  status="FAIL"
  if [[ "$rc" -eq 0 ]]; then
    status="PASS"
  fi
  if [[ -f "$metrics_file" ]]; then
    detected="$(metric_value_or_zero "$metrics_file" "detected_mutants")"
    relevant="$(metric_value_or_zero "$metrics_file" "relevant_mutants")"
    errors="$(metric_value_or_zero "$metrics_file" "errors")"
    if [[ "$relevant" =~ ^[0-9]+$ ]] && [[ "$detected" =~ ^[0-9]+$ ]] && [[ "$relevant" -gt 0 ]]; then
      coverage="$(awk -v d="$detected" -v r="$relevant" 'BEGIN { printf "%.2f", (100.0 * d) / r }')"
    fi
  fi

  append_summary_row "$SUMMARY_FILE" "$example_id" "$status" "$rc" "$detected" "$relevant" "$coverage" "$errors"
  if [[ "$rc" -ne 0 ]]; then
    overall_rc=1
  fi
done

cat "$SUMMARY_FILE"
exit "$overall_rc"

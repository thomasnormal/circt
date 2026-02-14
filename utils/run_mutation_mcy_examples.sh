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
  --example NAME           Example to run (repeatable; default: all known examples)
  --example-manifest FILE  Optional TSV mapping of examples to design/top and
                           per-example mutation policy:
                           example<TAB>design<TAB>top<TAB>[generate_count]
                           <TAB>[mutations_seed]<TAB>[mutations_modes]
                           <TAB>[mutations_mode_counts]
                           <TAB>[mutations_mode_weights]
                           <TAB>[mutations_profiles]<TAB>[mutations_cfg]
                           <TAB>[mutations_select]<TAB>[mutation_limit]
                           Optional fields accept '-' to inherit global values.
                           Relative design paths resolve under --examples-root
  --circt-mut PATH         circt-mut binary or command (default: auto-detect)
  --yosys PATH             yosys binary (default: yosys)
  --generate-count N       Mutations to generate in non-smoke mode (default: 32)
  --mutations-seed N       Seed used with --generate-mutations (default: 1)
  --mutations-modes CSV    Comma-separated mutate modes for auto-generation
  --mutations-mode-counts CSV
                           Comma-separated mode=count allocation for auto-generation
  --mutations-mode-weights CSV
                           Comma-separated mode=weight allocation for auto-generation
  --mutations-profiles CSV Comma-separated named mutate profiles for auto-generation
  --mutations-cfg CSV      Comma-separated KEY=VALUE mutate cfg entries
  --mutations-select CSV   Comma-separated mutate select expressions
  --mutation-limit N       Per-example mutation limit (default: 8)
  --min-detected N         Fail if detected mutants per example is below N
                           (default: 0)
  --min-coverage-percent P Fail if coverage percent per example is below P
                           (0-100, default: disabled)
  --max-errors N           Fail if reported errors per example exceed N
                           (default: disabled)
  --baseline-file FILE     Baseline summary TSV for drift checks/updates
  --update-baseline        Write current summary.tsv to --baseline-file
  --fail-on-diff           Fail on metric regression vs --baseline-file
  --require-policy-fingerprint-baseline
                           Require baseline rows to include policy_fingerprint
                           when evaluating --fail-on-diff
  --require-baseline-example-parity
                           Require baseline/current example-id parity when
                           evaluating --fail-on-diff
  --require-baseline-schema-version-match
                           Require baseline summary schema version to match
                           current summary schema when evaluating
                           --fail-on-diff
  --baseline-schema-version-file FILE
                           Optional baseline schema-version artifact used for
                           --require-baseline-schema-version-match (default:
                           <baseline-file>.schema-version when present)
  --summary-schema-version-file FILE
                           Output schema-version artifact for current summary
                           (default: <out-dir>/summary.schema-version)
  --strict-baseline-governance
                           Enable strict baseline governance bundle:
                           --require-policy-fingerprint-baseline +
                           --require-baseline-example-parity +
                           --require-baseline-schema-version-match +
                           --require-unique-example-selection
                           (requires --fail-on-diff)
  --require-unique-example-selection
                           Fail if --example contains duplicate example IDs
  --require-unique-summary-rows
                           Fail if summary.tsv contains duplicate example rows
  --drift-allowlist-file FILE
                           Optional allowlist for drift/summary-contract
                           regressions (requires --fail-on-diff or
                           --require-unique-summary-rows). Non-empty,
                           non-comment lines are
                           glob patterns over:
                             example::metric
                             example::metric::detail
  --fail-on-unused-drift-allowlist
                           Fail when allowlist entries are unused by current
                           drift candidates
  --drift-allowlist-unused-file FILE
                           Optional output path for unused allowlist entries
                           (default: OUT_DIR/drift-allowlist-unused.txt)
  --smoke                  Run smoke mode without yosys:
                           - use stub mutations file
                           - use identity fake create-mutated script
  --keep-work              Keep per-example helper scripts under out-dir/.work
  -h, --help               Show this help

Outputs:
  <out-dir>/summary.tsv    Aggregated example status/coverage summary
  <out-dir>/summary.schema-version
                           Current summary schema-version artifact
  <out-dir>/drift.tsv      Drift report (when --fail-on-diff)
  <out-dir>/drift-allowlist-unused.txt
                           Unused allowlist entries (when allowlist is set)
  <out-dir>/summary-contract.tsv
                           Summary contract report (when --require-unique-summary-rows)
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
MUTATIONS_SEED=1
MUTATIONS_MODES=""
MUTATIONS_MODE_COUNTS=""
MUTATIONS_MODE_WEIGHTS=""
MUTATIONS_PROFILES=""
MUTATIONS_CFG=""
MUTATIONS_SELECT=""
MUTATION_LIMIT=8
MIN_DETECTED=0
MIN_COVERAGE_PERCENT=""
MAX_ERRORS=""
BASELINE_FILE=""
BASELINE_SCHEMA_VERSION_FILE=""
BASELINE_SCHEMA_VERSION_FILE_EXPLICIT=0
SUMMARY_SCHEMA_VERSION_FILE=""
UPDATE_BASELINE=0
FAIL_ON_DIFF=0
REQUIRE_POLICY_FINGERPRINT_BASELINE=0
REQUIRE_BASELINE_EXAMPLE_PARITY=0
REQUIRE_BASELINE_SCHEMA_VERSION_MATCH=0
REQUIRE_UNIQUE_EXAMPLE_SELECTION=0
STRICT_BASELINE_GOVERNANCE=0
REQUIRE_UNIQUE_SUMMARY_ROWS=0
DRIFT_ALLOWLIST_FILE=""
FAIL_ON_UNUSED_DRIFT_ALLOWLIST=0
DRIFT_ALLOWLIST_UNUSED_FILE=""
SMOKE=0
KEEP_WORK=0
MUTATION_GENERATION_FLAGS_SEEN=0
EXAMPLE_MANIFEST=""
EXAMPLE_IDS=()
declare -A EXAMPLE_TO_DESIGN=()
declare -A EXAMPLE_TO_TOP=()
declare -A EXAMPLE_TO_GENERATE_COUNT=()
declare -A EXAMPLE_TO_MUTATIONS_SEED=()
declare -A EXAMPLE_TO_MUTATIONS_MODES=()
declare -A EXAMPLE_TO_MUTATIONS_MODE_COUNTS=()
declare -A EXAMPLE_TO_MUTATIONS_MODE_WEIGHTS=()
declare -A EXAMPLE_TO_MUTATIONS_PROFILES=()
declare -A EXAMPLE_TO_MUTATIONS_CFG=()
declare -A EXAMPLE_TO_MUTATIONS_SELECT=()
declare -A EXAMPLE_TO_MUTATION_LIMIT=()
declare -a AVAILABLE_EXAMPLES=()
declare -a DRIFT_ALLOW_PATTERNS=()
declare -A DRIFT_ALLOW_PATTERN_USED=()

SUMMARY_HEADER_V1=$'example\tstatus\texit_code\tdetected\trelevant\tcoverage_percent\terrors'
SUMMARY_HEADER_V2=$'example\tstatus\texit_code\tdetected\trelevant\tcoverage_percent\terrors\tpolicy_fingerprint'
CURRENT_SUMMARY_SCHEMA_VERSION="v2"
CURRENT_SUMMARY_HEADER="$SUMMARY_HEADER_V2"

is_pos_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonneg_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_nonneg_decimal() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

trim_whitespace() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s\n' "$s"
}


normalize_manifest_optional() {
  local s
  s="$(trim_whitespace "${1:-}")"
  if [[ "$s" == "-" ]]; then
    echo ""
  else
    echo "$s"
  fi
}

normalize_int_or_zero() {
  local raw="${1:-0}"
  if [[ "$raw" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$raw"
  else
    printf '0\n'
  fi
}

normalize_decimal_or_zero() {
  local raw="${1:-0}"
  if [[ "$raw" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf '%s\n' "$raw"
  else
    printf '0\n'
  fi
}

float_lt() {
  local lhs="${1:-0}"
  local rhs="${2:-0}"
  awk -v a="$lhs" -v b="$rhs" 'BEGIN { exit !(a + 0 < b + 0) }'
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


hash_stream_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
    return 0
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 | awk '{print $1}'
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import hashlib,sys;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'
    return 0
  fi
  cksum | awk '{print $1}'
}

hash_string_sha256() {
  local value="$1"
  printf '%s' "$value" | hash_stream_sha256
}

hash_file_sha256() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
    return 0
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$file"
    return 0
  fi
  cksum "$file" | awk '{print $1}'
}

load_drift_allowlist() {
  local file="$1"
  local raw_line=""
  local line=""
  DRIFT_ALLOW_PATTERNS=()
  DRIFT_ALLOW_PATTERN_USED=()
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%$'\r'}"
    line="$(trim_whitespace "$line")"
    if [[ -z "$line" ]]; then
      continue
    fi
    if [[ "${line:0:1}" == "#" ]]; then
      continue
    fi
    DRIFT_ALLOW_PATTERNS+=("$line")
  done < "$file"
}

is_drift_allowed() {
  local example="$1"
  local metric="$2"
  local detail="${3:-}"
  local token="${example}::${metric}"
  local token_detail="${example}::${metric}::${detail}"
  local idx=""
  local pattern=""
  for idx in "${!DRIFT_ALLOW_PATTERNS[@]}"; do
    pattern="${DRIFT_ALLOW_PATTERNS[$idx]}"
    if [[ "$token_detail" == $pattern ]] || [[ "$token" == $pattern ]]; then
      DRIFT_ALLOW_PATTERN_USED["$idx"]=1
      return 0
    fi
  done
  return 1
}

write_unused_drift_allowlist_report() {
  local out_file="$1"
  local idx=""
  local pattern=""
  local unused_count=0

  mkdir -p "$(dirname "$out_file")"
  : > "$out_file"

  for idx in "${!DRIFT_ALLOW_PATTERNS[@]}"; do
    if [[ -z "${DRIFT_ALLOW_PATTERN_USED[$idx]+x}" ]]; then
      pattern="${DRIFT_ALLOW_PATTERNS[$idx]}"
      printf '%s\n' "$pattern" >> "$out_file"
      unused_count=$((unused_count + 1))
    fi
  done

  printf '%s\n' "$unused_count"
}

register_example_mapping() {
  local id="$1"
  local design="$2"
  local top="$3"
  if [[ -n "${EXAMPLE_TO_DESIGN[$id]+x}" ]]; then
    return 1
  fi
  EXAMPLE_TO_DESIGN["$id"]="$design"
  EXAMPLE_TO_TOP["$id"]="$top"
  AVAILABLE_EXAMPLES+=("$id")
  return 0
}

reset_example_mappings() {
  EXAMPLE_TO_DESIGN=()
  EXAMPLE_TO_TOP=()
  EXAMPLE_TO_GENERATE_COUNT=()
  EXAMPLE_TO_MUTATIONS_SEED=()
  EXAMPLE_TO_MUTATIONS_MODES=()
  EXAMPLE_TO_MUTATIONS_MODE_COUNTS=()
  EXAMPLE_TO_MUTATIONS_MODE_WEIGHTS=()
  EXAMPLE_TO_MUTATIONS_PROFILES=()
  EXAMPLE_TO_MUTATIONS_CFG=()
  EXAMPLE_TO_MUTATIONS_SELECT=()
  EXAMPLE_TO_MUTATION_LIMIT=()
  AVAILABLE_EXAMPLES=()
}

load_default_examples() {
  reset_example_mappings
  register_example_mapping "bitcnt" "${EXAMPLES_ROOT}/bitcnt/bitcnt.v" "bitcnt"
  register_example_mapping "picorv32_primes" "${EXAMPLES_ROOT}/picorv32_primes/picorv32.v" "picorv32"
}

has_manifest_generation_overrides() {
  [[ ${#EXAMPLE_TO_GENERATE_COUNT[@]} -gt 0 ]] && return 0
  [[ ${#EXAMPLE_TO_MUTATIONS_SEED[@]} -gt 0 ]] && return 0
  [[ ${#EXAMPLE_TO_MUTATIONS_MODES[@]} -gt 0 ]] && return 0
  [[ ${#EXAMPLE_TO_MUTATIONS_MODE_COUNTS[@]} -gt 0 ]] && return 0
  [[ ${#EXAMPLE_TO_MUTATIONS_MODE_WEIGHTS[@]} -gt 0 ]] && return 0
  [[ ${#EXAMPLE_TO_MUTATIONS_PROFILES[@]} -gt 0 ]] && return 0
  [[ ${#EXAMPLE_TO_MUTATIONS_CFG[@]} -gt 0 ]] && return 0
  [[ ${#EXAMPLE_TO_MUTATIONS_SELECT[@]} -gt 0 ]] && return 0
  return 1
}

load_example_manifest() {
  local file="$1"
  local raw_line=""
  local line=""
  local line_no=0
  local example_id=""
  local design=""
  local top=""
  local generate_count_override=""
  local mutations_seed_override=""
  local mutations_modes_override=""
  local mutations_mode_counts_override=""
  local mutations_mode_weights_override=""
  local mutations_profiles_override=""
  local mutations_cfg_override=""
  local mutations_select_override=""
  local mutation_limit_override=""
  local extra=""
  local resolved_design=""

  reset_example_mappings

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line_no=$((line_no + 1))
    line="${raw_line%$'\r'}"
    line="$(trim_whitespace "$line")"
    if [[ -z "$line" || "${line:0:1}" == "#" ]]; then
      continue
    fi

    IFS=$'\t' read -r \
      example_id design top \
      generate_count_override mutations_seed_override mutations_modes_override \
      mutations_mode_counts_override mutations_mode_weights_override \
      mutations_profiles_override mutations_cfg_override mutations_select_override \
      mutation_limit_override extra <<< "$line"

    example_id="$(trim_whitespace "$example_id")"
    design="$(trim_whitespace "$design")"
    top="$(trim_whitespace "$top")"
    generate_count_override="$(normalize_manifest_optional "${generate_count_override:-}")"
    mutations_seed_override="$(normalize_manifest_optional "${mutations_seed_override:-}")"
    mutations_modes_override="$(normalize_manifest_optional "${mutations_modes_override:-}")"
    mutations_mode_counts_override="$(normalize_manifest_optional "${mutations_mode_counts_override:-}")"
    mutations_mode_weights_override="$(normalize_manifest_optional "${mutations_mode_weights_override:-}")"
    mutations_profiles_override="$(normalize_manifest_optional "${mutations_profiles_override:-}")"
    mutations_cfg_override="$(normalize_manifest_optional "${mutations_cfg_override:-}")"
    mutations_select_override="$(normalize_manifest_optional "${mutations_select_override:-}")"
    mutation_limit_override="$(normalize_manifest_optional "${mutation_limit_override:-}")"
    extra="$(trim_whitespace "${extra:-}")"

    if [[ -z "$example_id" || -z "$design" || -z "$top" || -n "$extra" ]]; then
      echo "Invalid example manifest row ${line_no} in ${file} (expected: example<TAB>design<TAB>top with up to 9 optional override columns)." >&2
      return 1
    fi

    if [[ -n "$generate_count_override" && ! "$generate_count_override" =~ ^[1-9][0-9]*$ ]]; then
      echo "Invalid generate_count override in manifest row ${line_no}: ${generate_count_override}" >&2
      return 1
    fi
    if [[ -n "$mutations_seed_override" && ! "$mutations_seed_override" =~ ^[0-9]+$ ]]; then
      echo "Invalid mutations_seed override in manifest row ${line_no}: ${mutations_seed_override}" >&2
      return 1
    fi
    if [[ -n "$mutation_limit_override" && ! "$mutation_limit_override" =~ ^[1-9][0-9]*$ ]]; then
      echo "Invalid mutation_limit override in manifest row ${line_no}: ${mutation_limit_override}" >&2
      return 1
    fi
    if [[ -n "$mutations_mode_counts_override" && -n "$mutations_mode_weights_override" ]]; then
      echo "Manifest row ${line_no} sets both mutations_mode_counts and mutations_mode_weights; choose one." >&2
      return 1
    fi

    if [[ "$design" == /* ]]; then
      resolved_design="$design"
    else
      resolved_design="${EXAMPLES_ROOT}/${design}"
    fi

    if ! register_example_mapping "$example_id" "$resolved_design" "$top"; then
      echo "Duplicate example id in manifest row ${line_no}: ${example_id}" >&2
      return 1
    fi

    if [[ -n "$generate_count_override" ]]; then
      EXAMPLE_TO_GENERATE_COUNT["$example_id"]="$generate_count_override"
    fi
    if [[ -n "$mutations_seed_override" ]]; then
      EXAMPLE_TO_MUTATIONS_SEED["$example_id"]="$mutations_seed_override"
    fi
    if [[ -n "$mutations_modes_override" ]]; then
      EXAMPLE_TO_MUTATIONS_MODES["$example_id"]="$mutations_modes_override"
    fi
    if [[ -n "$mutations_mode_counts_override" ]]; then
      EXAMPLE_TO_MUTATIONS_MODE_COUNTS["$example_id"]="$mutations_mode_counts_override"
    fi
    if [[ -n "$mutations_mode_weights_override" ]]; then
      EXAMPLE_TO_MUTATIONS_MODE_WEIGHTS["$example_id"]="$mutations_mode_weights_override"
    fi
    if [[ -n "$mutations_profiles_override" ]]; then
      EXAMPLE_TO_MUTATIONS_PROFILES["$example_id"]="$mutations_profiles_override"
    fi
    if [[ -n "$mutations_cfg_override" ]]; then
      EXAMPLE_TO_MUTATIONS_CFG["$example_id"]="$mutations_cfg_override"
    fi
    if [[ -n "$mutations_select_override" ]]; then
      EXAMPLE_TO_MUTATIONS_SELECT["$example_id"]="$mutations_select_override"
    fi
    if [[ -n "$mutation_limit_override" ]]; then
      EXAMPLE_TO_MUTATION_LIMIT["$example_id"]="$mutation_limit_override"
    fi
  done < "$file"

  if [[ ${#AVAILABLE_EXAMPLES[@]} -eq 0 ]]; then
    echo "Example manifest has no usable rows: ${file}" >&2
    return 1
  fi
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
  local policy_fingerprint="$9"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$example_id" "$status" "$exit_code" "$detected" "$relevant" "$coverage" "$errors" "$policy_fingerprint" \
    >> "$summary_file"
}

metrics_triplet_or_zero() {
  local file="$1"
  local key=""
  local value=""
  local detected="0"
  local relevant="0"
  local errors="0"
  local seen_detected=0
  local seen_relevant=0
  local seen_errors=0

  while IFS=$'\t' read -r key value _rest; do
    case "$key" in
      detected_mutants)
        if [[ "$seen_detected" -eq 0 ]]; then
          detected="${value:-0}"
          seen_detected=1
        fi
        ;;
      relevant_mutants)
        if [[ "$seen_relevant" -eq 0 ]]; then
          relevant="${value:-0}"
          seen_relevant=1
        fi
        ;;
      errors)
        if [[ "$seen_errors" -eq 0 ]]; then
          errors="${value:-0}"
          seen_errors=1
        fi
        ;;
    esac
  done < "$file"

  printf '%s\t%s\t%s\n' "$detected" "$relevant" "$errors"
}

append_drift_row() {
  local drift_file="$1"
  local example_id="$2"
  local metric="$3"
  local baseline_value="$4"
  local current_value="$5"
  local outcome="$6"
  local detail="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$example_id" "$metric" "$baseline_value" "$current_value" "$outcome" "$detail" \
    >> "$drift_file"
}

append_drift_candidate() {
  local drift_file="$1"
  local example_id="$2"
  local metric="$3"
  local baseline_value="$4"
  local current_value="$5"
  local detail="$6"
  if is_drift_allowed "$example_id" "$metric" "$detail"; then
    append_drift_row "$drift_file" "$example_id" "$metric" "$baseline_value" "$current_value" "allowed" "$detail"
    return 0
  fi
  append_drift_row "$drift_file" "$example_id" "$metric" "$baseline_value" "$current_value" "regression" "$detail"
  return 1
}

summary_schema_version_from_header() {
  local header_line="$1"
  if [[ "$header_line" == "$SUMMARY_HEADER_V2" ]]; then
    printf 'v2\n'
    return 0
  fi
  if [[ "$header_line" == "$SUMMARY_HEADER_V1" ]]; then
    printf 'v1\n'
    return 0
  fi
  printf 'unknown\n'
}

summary_schema_version_from_metadata() {
  local schema_version_raw="$1"
  local schema_version=""
  schema_version="$(trim_whitespace "$schema_version_raw")"
  if [[ "$schema_version" =~ ^v[0-9]+$ ]]; then
    printf '%s\n' "$schema_version"
    return 0
  fi
  printf 'unknown\n'
}

summary_schema_version_for_artifacts() {
  local summary_file="$1"
  local schema_file="${2:-}"
  local schema_line=""
  local header_line=""

  if [[ -n "$schema_file" && -f "$schema_file" ]]; then
    if ! IFS= read -r schema_line < "$schema_file"; then
      printf 'unknown\n'
      return 0
    fi
    summary_schema_version_from_metadata "$schema_line"
    return 0
  fi

  if ! IFS= read -r header_line < "$summary_file"; then
    printf 'unknown\n'
    return 0
  fi
  summary_schema_version_from_header "$header_line"
}
sanitize_contract_field() {
  local value="$1"
  value="${value//$'\t'/ }"
  value="${value//$'\n'/ }"
  printf '%s\n' "$value"
}

evaluate_summary_contract() {
  local summary_file="$1"
  local contract_file="$2"
  local regressions=0
  local summary_example="__summary__"
  local expected_header="$CURRENT_SUMMARY_HEADER"
  local header_line=""
  local example=""
  local status=""
  local exit_code=""
  local detected=""
  local relevant=""
  local coverage=""
  local errors=""
  local policy=""
  local extra=""
  local -A summary_seen=()
  local -A summary_duplicate_seen=()
  local -a summary_duplicate_examples=()

  printf 'example\tmetric\texpected\tcurrent\toutcome\tdetail\n' > "$contract_file"

  if ! IFS= read -r header_line < "$summary_file"; then
    if ! append_drift_candidate "$contract_file" "$summary_example" "header" "$(sanitize_contract_field "$expected_header")" "missing" "missing_header"; then
      regressions=$((regressions + 1))
    fi
    if [[ "$regressions" -gt 0 ]]; then
      return 1
    fi
    return 0
  fi
  if [[ "$header_line" != "$expected_header" ]]; then
    if ! append_drift_candidate "$contract_file" "$summary_example" "header" "$(sanitize_contract_field "$expected_header")" "$(sanitize_contract_field "$header_line")" "header_mismatch"; then
      regressions=$((regressions + 1))
    fi
  fi

  while IFS=$'\t' read -r example status exit_code detected relevant coverage errors policy extra; do
    [[ "$example" == "example" ]] && continue
    if [[ -z "$example" ]]; then
      continue
    fi

    if [[ -n "$extra" ]]; then
      if ! append_drift_candidate "$contract_file" "$example" "column_count" "8" "9+" "invalid_column_count"; then
        regressions=$((regressions + 1))
      fi
    fi

    if [[ "$status" != "PASS" && "$status" != "FAIL" ]]; then
      if ! append_drift_candidate "$contract_file" "$example" "status" "PASS_or_FAIL" "$(sanitize_contract_field "$status")" "invalid_status"; then
        regressions=$((regressions + 1))
      fi
    fi

    if ! is_nonneg_int "$exit_code"; then
      if ! append_drift_candidate "$contract_file" "$example" "exit_code" "nonneg_int" "$(sanitize_contract_field "$exit_code")" "invalid_exit_code"; then
        regressions=$((regressions + 1))
      fi
    fi
    if ! is_nonneg_int "$detected"; then
      if ! append_drift_candidate "$contract_file" "$example" "detected_mutants" "nonneg_int" "$(sanitize_contract_field "$detected")" "invalid_detected"; then
        regressions=$((regressions + 1))
      fi
    fi
    if ! is_nonneg_int "$relevant"; then
      if ! append_drift_candidate "$contract_file" "$example" "relevant_mutants" "nonneg_int" "$(sanitize_contract_field "$relevant")" "invalid_relevant"; then
        regressions=$((regressions + 1))
      fi
    fi
    if ! is_nonneg_int "$errors"; then
      if ! append_drift_candidate "$contract_file" "$example" "errors" "nonneg_int" "$(sanitize_contract_field "$errors")" "invalid_errors"; then
        regressions=$((regressions + 1))
      fi
    fi

    if [[ "$coverage" == "-" ]]; then
      if is_nonneg_int "$relevant" && [[ "$relevant" -gt 0 ]]; then
        if ! append_drift_candidate "$contract_file" "$example" "coverage_percent" "numeric" "-" "coverage_missing_with_relevant"; then
          regressions=$((regressions + 1))
        fi
      fi
    else
      if ! is_nonneg_decimal "$coverage"; then
        if ! append_drift_candidate "$contract_file" "$example" "coverage_percent" "nonneg_decimal_or_dash" "$(sanitize_contract_field "$coverage")" "invalid_coverage_format"; then
          regressions=$((regressions + 1))
        fi
      elif ! awk -v v="$coverage" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
        if ! append_drift_candidate "$contract_file" "$example" "coverage_percent" "0_to_100" "$(sanitize_contract_field "$coverage")" "invalid_coverage_range"; then
          regressions=$((regressions + 1))
        fi
      elif is_nonneg_int "$relevant" && [[ "$relevant" -eq 0 ]]; then
        if ! append_drift_candidate "$contract_file" "$example" "coverage_percent" "-" "$(sanitize_contract_field "$coverage")" "coverage_present_with_zero_relevant"; then
          regressions=$((regressions + 1))
        fi
      fi
    fi

    if [[ -z "$policy" ]]; then
      if ! append_drift_candidate "$contract_file" "$example" "policy_fingerprint" "nonempty" "missing" "missing_policy_fingerprint"; then
        regressions=$((regressions + 1))
      fi
    fi

    if [[ -n "${summary_seen[$example]+x}" ]]; then
      if [[ -z "${summary_duplicate_seen[$example]+x}" ]]; then
        summary_duplicate_seen["$example"]=1
        summary_duplicate_examples+=("$example")
      fi
      continue
    fi
    summary_seen["$example"]=1
  done < "$summary_file"

  for example in "${summary_duplicate_examples[@]}"; do
    if ! append_drift_candidate "$contract_file" "$example" "row" "single_row" "duplicate_rows" "duplicate_current_row"; then
      regressions=$((regressions + 1))
    fi
  done

  if [[ "$regressions" -gt 0 ]]; then
    return 1
  fi
  return 0
}
evaluate_summary_drift() {
  local baseline_file="$1"
  local summary_file="$2"
  local drift_file="$3"
  local require_policy_fingerprint_baseline="${4:-0}"
  local require_baseline_example_parity="${5:-0}"
  local require_baseline_schema_version_match="${6:-0}"
  local baseline_schema_version_file="${7:-}"
  local summary_schema_version_file="${8:-}"
  local regressions=0
  local baseline_example=""
  local _status=""
  local _exit=""
  local _detected=""
  local _relevant=""
  local _coverage=""
  local _errors=""
  local _policy=""
  local baseline_row=""
  local summary_example=""
  local -A summary_examples_seen=()
  local -A summary_duplicate_seen=()
  local -A baseline_rows=()
  local -A baseline_duplicate_seen=()
  local -a summary_duplicate_examples=()
  local -a baseline_order=()
  local -a baseline_duplicate_examples=()

  printf 'example\tmetric\tbaseline\tcurrent\toutcome\tdetail\n' > "$drift_file"

  local baseline_schema_version=""
  local summary_schema_version=""
  summary_schema_version="$(summary_schema_version_for_artifacts "$summary_file" "$summary_schema_version_file")"
  baseline_schema_version="$(summary_schema_version_for_artifacts "$baseline_file" "$baseline_schema_version_file")"

  if [[ "$require_baseline_schema_version_match" -eq 1 ]]; then
    if [[ "$baseline_schema_version" != "$summary_schema_version" ]]; then
      if ! append_drift_candidate "$drift_file" "__baseline__" "baseline_schema_version" "$baseline_schema_version" "$summary_schema_version" "baseline_schema_version_mismatch"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "__baseline__" "baseline_schema_version" "$baseline_schema_version" "$summary_schema_version" "ok" ""
    fi
  fi

  while IFS=$'\t' read -r baseline_example _status _exit _detected _relevant _coverage _errors _policy; do
    [[ "$baseline_example" == "example" ]] && continue
    if [[ -z "$baseline_example" ]]; then
      continue
    fi
    if [[ -n "${baseline_rows[$baseline_example]+x}" ]]; then
      if [[ -z "${baseline_duplicate_seen[$baseline_example]+x}" ]]; then
        baseline_duplicate_seen["$baseline_example"]=1
        baseline_duplicate_examples+=("$baseline_example")
      fi
      continue
    fi
    baseline_rows["$baseline_example"]="$baseline_example"$'\t'"${_status}"$'\t'"${_exit}"$'\t'"${_detected}"$'\t'"${_relevant}"$'\t'"${_coverage}"$'\t'"${_errors}"$'\t'"${_policy}"
    baseline_order+=("$baseline_example")
  done < "$baseline_file"

  while IFS=$'\t' read -r example status exit_code detected relevant coverage errors policy_fingerprint; do
    [[ "$example" == "example" ]] && continue
    if [[ -z "$example" ]]; then
      continue
    fi

    if [[ -n "${summary_examples_seen[$example]+x}" ]]; then
      if [[ -z "${summary_duplicate_seen[$example]+x}" ]]; then
        summary_duplicate_seen["$example"]=1
        summary_duplicate_examples+=("$example")
      fi
      continue
    fi
    summary_examples_seen["$example"]=1

    detected="$(normalize_int_or_zero "$detected")"
    relevant="$(normalize_int_or_zero "$relevant")"
    errors="$(normalize_int_or_zero "$errors")"
    policy_fingerprint="$(trim_whitespace "${policy_fingerprint:-}")"
    local coverage_num
    coverage_num="$(normalize_decimal_or_zero "$coverage")"

    baseline_row="${baseline_rows[$example]-}"
    if [[ -z "$baseline_row" ]]; then
      if ! append_drift_candidate "$drift_file" "$example" "row" "present" "missing" "missing_baseline_row"; then
        regressions=$((regressions + 1))
      fi
      continue
    fi

    local _be _bs _bexit _bd _br _bc _berr _bpolicy
    IFS=$'\t' read -r _be _bs _bexit _bd _br _bc _berr _bpolicy <<< "$baseline_row"

    _bd="$(normalize_int_or_zero "${_bd:-0}")"
    _br="$(normalize_int_or_zero "${_br:-0}")"
    _berr="$(normalize_int_or_zero "${_berr:-0}")"
    _bpolicy="$(trim_whitespace "${_bpolicy:-}")"
    local base_cov_num
    base_cov_num="$(normalize_decimal_or_zero "${_bc:-0}")"

    if [[ "${_bs:-}" == "PASS" && "$status" != "PASS" ]]; then
      if ! append_drift_candidate "$drift_file" "$example" "status" "${_bs:-}" "$status" "status_regressed"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "$example" "status" "${_bs:-}" "$status" "ok" ""
    fi

    if [[ "$detected" -lt "$_bd" ]]; then
      if ! append_drift_candidate "$drift_file" "$example" "detected_mutants" "$_bd" "$detected" "detected_decreased"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "$example" "detected_mutants" "$_bd" "$detected" "ok" ""
    fi

    if float_lt "$coverage_num" "$base_cov_num"; then
      if ! append_drift_candidate "$drift_file" "$example" "coverage_percent" "$base_cov_num" "$coverage_num" "coverage_decreased"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "$example" "coverage_percent" "$base_cov_num" "$coverage_num" "ok" ""
    fi

    if [[ "$errors" -gt "$_berr" ]]; then
      if ! append_drift_candidate "$drift_file" "$example" "errors" "$_berr" "$errors" "errors_increased"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "$example" "errors" "$_berr" "$errors" "ok" ""
    fi

    if [[ "$relevant" -lt "$_br" ]]; then
      if ! append_drift_candidate "$drift_file" "$example" "relevant_mutants" "$_br" "$relevant" "relevant_decreased"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "$example" "relevant_mutants" "$_br" "$relevant" "ok" ""
    fi

    if [[ -n "$_bpolicy" ]]; then
      if [[ -z "$policy_fingerprint" ]]; then
        if ! append_drift_candidate "$drift_file" "$example" "policy_fingerprint" "$_bpolicy" "missing" "policy_missing"; then
          regressions=$((regressions + 1))
        fi
      elif [[ "$policy_fingerprint" != "$_bpolicy" ]]; then
        if ! append_drift_candidate "$drift_file" "$example" "policy_fingerprint" "$_bpolicy" "$policy_fingerprint" "policy_changed"; then
          regressions=$((regressions + 1))
        fi
      else
        append_drift_row "$drift_file" "$example" "policy_fingerprint" "$_bpolicy" "$policy_fingerprint" "ok" ""
      fi
    else
      if [[ "$require_policy_fingerprint_baseline" -eq 1 ]]; then
        if ! append_drift_candidate "$drift_file" "$example" "policy_fingerprint" "-" "${policy_fingerprint:-missing}" "baseline_missing_policy_fingerprint"; then
          regressions=$((regressions + 1))
        fi
      else
        append_drift_row "$drift_file" "$example" "policy_fingerprint" "-" "${policy_fingerprint:-missing}" "ok" "baseline_missing_policy_fingerprint"
      fi
    fi
  done < "$summary_file"

  if [[ "$require_baseline_example_parity" -eq 1 ]]; then
    for baseline_example in "${baseline_order[@]}"; do
      if [[ -z "${summary_examples_seen[$baseline_example]+x}" ]]; then
        if ! append_drift_candidate "$drift_file" "$baseline_example" "row" "present" "missing" "missing_current_row"; then
          regressions=$((regressions + 1))
        fi
      fi
    done
  fi

  for baseline_example in "${baseline_duplicate_examples[@]}"; do
    if ! append_drift_candidate "$drift_file" "$baseline_example" "row" "single_row" "duplicate_rows" "duplicate_baseline_row"; then
      regressions=$((regressions + 1))
    fi
  done

  for summary_example in "${summary_duplicate_examples[@]}"; do
    if ! append_drift_candidate "$drift_file" "$summary_example" "row" "single_row" "duplicate_rows" "duplicate_current_row"; then
      regressions=$((regressions + 1))
    fi
  done

  if [[ "$regressions" -gt 0 ]]; then
    return 1
  fi
  return 0
}


ensure_unique_example_selection() {
  local -A seen=()
  local example_id=""
  for example_id in "${EXAMPLE_IDS[@]}"; do
    if [[ -n "${seen[$example_id]+x}" ]]; then
      echo "Duplicate --example selection: ${example_id}" >&2
      return 1
    fi
    seen["$example_id"]=1
  done
  return 0
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
    --example-manifest)
      EXAMPLE_MANIFEST="$2"
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
    --mutations-seed)
      MUTATIONS_SEED="$2"
      MUTATION_GENERATION_FLAGS_SEEN=1
      shift 2
      ;;
    --mutations-modes)
      MUTATIONS_MODES="$2"
      MUTATION_GENERATION_FLAGS_SEEN=1
      shift 2
      ;;
    --mutations-mode-counts)
      MUTATIONS_MODE_COUNTS="$2"
      MUTATION_GENERATION_FLAGS_SEEN=1
      shift 2
      ;;
    --mutations-mode-weights)
      MUTATIONS_MODE_WEIGHTS="$2"
      MUTATION_GENERATION_FLAGS_SEEN=1
      shift 2
      ;;
    --mutations-profiles)
      MUTATIONS_PROFILES="$2"
      MUTATION_GENERATION_FLAGS_SEEN=1
      shift 2
      ;;
    --mutations-cfg)
      MUTATIONS_CFG="$2"
      MUTATION_GENERATION_FLAGS_SEEN=1
      shift 2
      ;;
    --mutations-select)
      MUTATIONS_SELECT="$2"
      MUTATION_GENERATION_FLAGS_SEEN=1
      shift 2
      ;;
    --mutation-limit)
      MUTATION_LIMIT="$2"
      shift 2
      ;;
    --min-detected)
      MIN_DETECTED="$2"
      shift 2
      ;;
    --min-coverage-percent)
      MIN_COVERAGE_PERCENT="$2"
      shift 2
      ;;
    --max-errors)
      MAX_ERRORS="$2"
      shift 2
      ;;
    --baseline-file)
      BASELINE_FILE="$2"
      shift 2
      ;;
    --update-baseline)
      UPDATE_BASELINE=1
      shift
      ;;
    --fail-on-diff)
      FAIL_ON_DIFF=1
      shift
      ;;
    --require-policy-fingerprint-baseline)
      REQUIRE_POLICY_FINGERPRINT_BASELINE=1
      shift
      ;;
    --require-baseline-example-parity)
      REQUIRE_BASELINE_EXAMPLE_PARITY=1
      shift
      ;;
    --require-baseline-schema-version-match)
      REQUIRE_BASELINE_SCHEMA_VERSION_MATCH=1
      shift
      ;;
    --summary-schema-version-file)
      SUMMARY_SCHEMA_VERSION_FILE="$2"
      shift 2
      ;;
    --baseline-schema-version-file)
      BASELINE_SCHEMA_VERSION_FILE="$2"
      BASELINE_SCHEMA_VERSION_FILE_EXPLICIT=1
      shift 2
      ;;
    --strict-baseline-governance)
      STRICT_BASELINE_GOVERNANCE=1
      shift
      ;;
    --require-unique-example-selection)
      REQUIRE_UNIQUE_EXAMPLE_SELECTION=1
      shift
      ;;
    --require-unique-summary-rows)
      REQUIRE_UNIQUE_SUMMARY_ROWS=1
      shift
      ;;
    --drift-allowlist-file)
      DRIFT_ALLOWLIST_FILE="$2"
      shift 2
      ;;
    --fail-on-unused-drift-allowlist)
      FAIL_ON_UNUSED_DRIFT_ALLOWLIST=1
      shift
      ;;
    --drift-allowlist-unused-file)
      DRIFT_ALLOWLIST_UNUSED_FILE="$2"
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

if [[ "$STRICT_BASELINE_GOVERNANCE" -eq 1 ]]; then
  REQUIRE_POLICY_FINGERPRINT_BASELINE=1
  REQUIRE_BASELINE_EXAMPLE_PARITY=1
  REQUIRE_BASELINE_SCHEMA_VERSION_MATCH=1
  REQUIRE_UNIQUE_EXAMPLE_SELECTION=1
fi

if ! is_pos_int "$GENERATE_COUNT"; then
  echo "--generate-count must be a positive integer: $GENERATE_COUNT" >&2
  exit 1
fi
if ! is_nonneg_int "$MUTATIONS_SEED"; then
  echo "--mutations-seed must be a non-negative integer: $MUTATIONS_SEED" >&2
  exit 1
fi
if [[ -n "$MUTATIONS_MODE_COUNTS" && -n "$MUTATIONS_MODE_WEIGHTS" ]]; then
  echo "Use either --mutations-mode-counts or --mutations-mode-weights, not both." >&2
  exit 1
fi
if ! is_pos_int "$MUTATION_LIMIT"; then
  echo "--mutation-limit must be a positive integer: $MUTATION_LIMIT" >&2
  exit 1
fi
if ! is_nonneg_int "$MIN_DETECTED"; then
  echo "--min-detected must be a non-negative integer: $MIN_DETECTED" >&2
  exit 1
fi
if [[ -n "$MIN_COVERAGE_PERCENT" ]]; then
  if ! is_nonneg_decimal "$MIN_COVERAGE_PERCENT"; then
    echo "--min-coverage-percent must be numeric in range [0,100]: $MIN_COVERAGE_PERCENT" >&2
    exit 1
  fi
  if ! awk -v v="$MIN_COVERAGE_PERCENT" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
    echo "--min-coverage-percent must be numeric in range [0,100]: $MIN_COVERAGE_PERCENT" >&2
    exit 1
  fi
fi
if [[ -n "$MAX_ERRORS" ]] && ! is_nonneg_int "$MAX_ERRORS"; then
  echo "--max-errors must be a non-negative integer: $MAX_ERRORS" >&2
  exit 1
fi
if [[ "$UPDATE_BASELINE" -eq 1 || "$FAIL_ON_DIFF" -eq 1 ]]; then
  if [[ -z "$BASELINE_FILE" ]]; then
    echo "--baseline-file is required with --update-baseline or --fail-on-diff" >&2
    exit 1
  fi
fi
if [[ "$UPDATE_BASELINE" -eq 1 && "$FAIL_ON_DIFF" -eq 1 ]]; then
  echo "Use either --update-baseline or --fail-on-diff, not both." >&2
  exit 1
fi
if [[ -z "$SUMMARY_SCHEMA_VERSION_FILE" ]]; then
  SUMMARY_SCHEMA_VERSION_FILE="${OUT_DIR}/summary.schema-version"
fi
if [[ -z "$BASELINE_SCHEMA_VERSION_FILE" && -n "$BASELINE_FILE" ]]; then
  BASELINE_SCHEMA_VERSION_FILE="${BASELINE_FILE}.schema-version"
fi
if [[ "$BASELINE_SCHEMA_VERSION_FILE_EXPLICIT" -eq 1 && "$FAIL_ON_DIFF" -eq 1 && "$REQUIRE_BASELINE_SCHEMA_VERSION_MATCH" -eq 1 ]]; then
  if [[ ! -f "$BASELINE_SCHEMA_VERSION_FILE" ]]; then
    echo "Baseline schema-version file not found: $BASELINE_SCHEMA_VERSION_FILE" >&2
    exit 1
  fi
  if [[ ! -r "$BASELINE_SCHEMA_VERSION_FILE" ]]; then
    echo "Baseline schema-version file not readable: $BASELINE_SCHEMA_VERSION_FILE" >&2
    exit 1
  fi
fi

if [[ "$STRICT_BASELINE_GOVERNANCE" -eq 1 && "$FAIL_ON_DIFF" -ne 1 ]]; then
  echo "--strict-baseline-governance requires --fail-on-diff" >&2
  exit 1
fi
if [[ "$REQUIRE_POLICY_FINGERPRINT_BASELINE" -eq 1 && "$FAIL_ON_DIFF" -ne 1 ]]; then
  echo "--require-policy-fingerprint-baseline requires --fail-on-diff" >&2
  exit 1
fi
if [[ "$REQUIRE_BASELINE_EXAMPLE_PARITY" -eq 1 && "$FAIL_ON_DIFF" -ne 1 ]]; then
  echo "--require-baseline-example-parity requires --fail-on-diff" >&2
  exit 1
fi
if [[ "$REQUIRE_BASELINE_SCHEMA_VERSION_MATCH" -eq 1 && "$FAIL_ON_DIFF" -ne 1 ]]; then
  echo "--require-baseline-schema-version-match requires --fail-on-diff" >&2
  exit 1
fi
if [[ -n "$DRIFT_ALLOWLIST_FILE" && "$FAIL_ON_DIFF" -ne 1 && "$REQUIRE_UNIQUE_SUMMARY_ROWS" -ne 1 ]]; then
  echo "--drift-allowlist-file requires --fail-on-diff or --require-unique-summary-rows" >&2
  exit 1
fi
if [[ "$FAIL_ON_UNUSED_DRIFT_ALLOWLIST" -eq 1 && -z "$DRIFT_ALLOWLIST_FILE" ]]; then
  echo "--fail-on-unused-drift-allowlist requires --drift-allowlist-file" >&2
  exit 1
fi
if [[ -n "$DRIFT_ALLOWLIST_UNUSED_FILE" && -z "$DRIFT_ALLOWLIST_FILE" ]]; then
  echo "--drift-allowlist-unused-file requires --drift-allowlist-file" >&2
  exit 1
fi
if [[ "$FAIL_ON_DIFF" -eq 1 ]]; then
  if [[ ! -f "$BASELINE_FILE" ]]; then
    echo "Baseline file not found: $BASELINE_FILE" >&2
    exit 1
  fi
  if [[ ! -r "$BASELINE_FILE" ]]; then
    echo "Baseline file not readable: $BASELINE_FILE" >&2
    exit 1
  fi
fi
if [[ -n "$DRIFT_ALLOWLIST_FILE" ]]; then
  if [[ ! -f "$DRIFT_ALLOWLIST_FILE" ]]; then
    echo "Drift allowlist file not found: $DRIFT_ALLOWLIST_FILE" >&2
    exit 1
  fi
  if [[ ! -r "$DRIFT_ALLOWLIST_FILE" ]]; then
    echo "Drift allowlist file not readable: $DRIFT_ALLOWLIST_FILE" >&2
    exit 1
  fi
  load_drift_allowlist "$DRIFT_ALLOWLIST_FILE"
fi

if [[ -n "$EXAMPLE_MANIFEST" ]]; then
  if [[ ! -f "$EXAMPLE_MANIFEST" ]]; then
    echo "Example manifest file not found: $EXAMPLE_MANIFEST" >&2
    exit 1
  fi
  if [[ ! -r "$EXAMPLE_MANIFEST" ]]; then
    echo "Example manifest file not readable: $EXAMPLE_MANIFEST" >&2
    exit 1
  fi
  load_example_manifest "$EXAMPLE_MANIFEST"
else
  load_default_examples
fi

if [[ "$SMOKE" -eq 1 ]]; then
  if [[ "$MUTATION_GENERATION_FLAGS_SEEN" -eq 1 ]] || has_manifest_generation_overrides; then
    echo "Mutation generation options (--mutations-*) require non-smoke mode." >&2
    exit 1
  fi
fi

if [[ ${#EXAMPLE_IDS[@]} -eq 0 ]]; then
  EXAMPLE_IDS=("${AVAILABLE_EXAMPLES[@]}")
fi

for example_id in "${EXAMPLE_IDS[@]}"; do
  if [[ -z "${EXAMPLE_TO_DESIGN[$example_id]+x}" ]]; then
    echo "Unknown --example value: $example_id" >&2
    exit 1
  fi
done

if [[ "$REQUIRE_UNIQUE_EXAMPLE_SELECTION" -eq 1 ]]; then
  ensure_unique_example_selection
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
if [[ -n "$DRIFT_ALLOWLIST_FILE" && -z "$DRIFT_ALLOWLIST_UNUSED_FILE" ]]; then
  DRIFT_ALLOWLIST_UNUSED_FILE="${OUT_DIR}/drift-allowlist-unused.txt"
fi
WORK_ROOT="${OUT_DIR}/.work"
mkdir -p "$WORK_ROOT"
if [[ "$KEEP_WORK" -ne 1 ]]; then
  trap 'rm -rf "$WORK_ROOT"' EXIT
fi

SUMMARY_FILE="${OUT_DIR}/summary.tsv"
printf '%s\n' "$CURRENT_SUMMARY_HEADER" > "$SUMMARY_FILE"

overall_rc=0
mkdir -p "$(dirname "$SUMMARY_SCHEMA_VERSION_FILE")"
printf '%s\n' "$CURRENT_SUMMARY_SCHEMA_VERSION" > "$SUMMARY_SCHEMA_VERSION_FILE"

for example_id in "${EXAMPLE_IDS[@]}"; do
  design="${EXAMPLE_TO_DESIGN[$example_id]}"
  top="${EXAMPLE_TO_TOP[$example_id]}"
  example_generate_count="$GENERATE_COUNT"
  example_mutations_seed="$MUTATIONS_SEED"
  example_mutations_modes="$MUTATIONS_MODES"
  example_mutations_mode_counts="$MUTATIONS_MODE_COUNTS"
  example_mutations_mode_weights="$MUTATIONS_MODE_WEIGHTS"
  example_mutations_profiles="$MUTATIONS_PROFILES"
  example_mutations_cfg="$MUTATIONS_CFG"
  example_mutations_select="$MUTATIONS_SELECT"
  example_mutation_limit="$MUTATION_LIMIT"

  if [[ -n "${EXAMPLE_TO_GENERATE_COUNT[$example_id]+x}" ]]; then
    example_generate_count="${EXAMPLE_TO_GENERATE_COUNT[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_MUTATIONS_SEED[$example_id]+x}" ]]; then
    example_mutations_seed="${EXAMPLE_TO_MUTATIONS_SEED[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_MUTATIONS_MODES[$example_id]+x}" ]]; then
    example_mutations_modes="${EXAMPLE_TO_MUTATIONS_MODES[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_MUTATIONS_MODE_COUNTS[$example_id]+x}" ]]; then
    example_mutations_mode_counts="${EXAMPLE_TO_MUTATIONS_MODE_COUNTS[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_MUTATIONS_MODE_WEIGHTS[$example_id]+x}" ]]; then
    example_mutations_mode_weights="${EXAMPLE_TO_MUTATIONS_MODE_WEIGHTS[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_MUTATIONS_PROFILES[$example_id]+x}" ]]; then
    example_mutations_profiles="${EXAMPLE_TO_MUTATIONS_PROFILES[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_MUTATIONS_CFG[$example_id]+x}" ]]; then
    example_mutations_cfg="${EXAMPLE_TO_MUTATIONS_CFG[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_MUTATIONS_SELECT[$example_id]+x}" ]]; then
    example_mutations_select="${EXAMPLE_TO_MUTATIONS_SELECT[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_MUTATION_LIMIT[$example_id]+x}" ]]; then
    example_mutation_limit="${EXAMPLE_TO_MUTATION_LIMIT[$example_id]}"
  fi

  if [[ -n "$example_mutations_mode_counts" && -n "$example_mutations_mode_weights" ]]; then
    echo "Resolved mutation mode allocation conflict for ${example_id}: both mode-counts and mode-weights are set." >&2
    exit 1
  fi

  if [[ ! -f "$design" ]]; then
    echo "Missing example design for ${example_id}: $design" >&2
    exit 1
  fi

  design_content_hash="$(hash_file_sha256 "$design")"
  policy_fingerprint_input="${example_id}"$'\n'"${top}"$'\n'"${design_content_hash}"$'\n'"${example_generate_count}"$'\n'"${example_mutations_seed}"$'\n'"${example_mutations_modes}"$'\n'"${example_mutations_mode_counts}"$'\n'"${example_mutations_mode_weights}"$'\n'"${example_mutations_profiles}"$'\n'"${example_mutations_cfg}"$'\n'"${example_mutations_select}"$'\n'"${example_mutation_limit}"$'\n'"${SMOKE}"
  policy_fingerprint="$(hash_string_sha256 "$policy_fingerprint_input")"

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
    --mutation-limit "$example_mutation_limit"
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
      --generate-mutations "$example_generate_count"
      --mutations-top "$top"
      --mutations-yosys "$YOSYS_RESOLVED"
      --mutations-seed "$example_mutations_seed"
    )
    if [[ -n "$example_mutations_modes" ]]; then
      cmd+=(--mutations-modes "$example_mutations_modes")
    fi
    if [[ -n "$example_mutations_mode_counts" ]]; then
      cmd+=(--mutations-mode-counts "$example_mutations_mode_counts")
    fi
    if [[ -n "$example_mutations_mode_weights" ]]; then
      cmd+=(--mutations-mode-weights "$example_mutations_mode_weights")
    fi
    if [[ -n "$example_mutations_profiles" ]]; then
      cmd+=(--mutations-profiles "$example_mutations_profiles")
    fi
    if [[ -n "$example_mutations_cfg" ]]; then
      cmd+=(--mutations-cfg "$example_mutations_cfg")
    fi
    if [[ -n "$example_mutations_select" ]]; then
      cmd+=(--mutations-select "$example_mutations_select")
    fi
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
  coverage_for_gate="0"
  status="FAIL"
  if [[ "$rc" -eq 0 ]]; then
    status="PASS"
  fi
  if [[ -f "$metrics_file" ]]; then
    IFS=$'\t' read -r detected relevant errors <<< "$(metrics_triplet_or_zero "$metrics_file")"
    detected="$(normalize_int_or_zero "$detected")"
    relevant="$(normalize_int_or_zero "$relevant")"
    errors="$(normalize_int_or_zero "$errors")"
    if [[ "$relevant" -gt 0 ]]; then
      coverage="$(awk -v d="$detected" -v r="$relevant" 'BEGIN { printf "%.2f", (100.0 * d) / r }')"
      coverage_for_gate="$coverage"
    fi
  fi

  gate_failure=""
  if [[ "$detected" -lt "$MIN_DETECTED" ]]; then
    gate_failure="detected<${MIN_DETECTED}"
  fi
  if [[ -n "$MIN_COVERAGE_PERCENT" ]]; then
    if float_lt "$coverage_for_gate" "$MIN_COVERAGE_PERCENT"; then
      if [[ -n "$gate_failure" ]]; then
        gate_failure+=","
      fi
      gate_failure+="coverage<${MIN_COVERAGE_PERCENT}"
    fi
  fi
  if [[ -n "$MAX_ERRORS" ]] && [[ "$errors" -gt "$MAX_ERRORS" ]]; then
    if [[ -n "$gate_failure" ]]; then
      gate_failure+=","
    fi
    gate_failure+="errors>${MAX_ERRORS}"
  fi
  if [[ -n "$gate_failure" ]]; then
    status="FAIL"
    overall_rc=1
    echo "Gate failure (${example_id}): ${gate_failure} (detected=${detected} relevant=${relevant} coverage=${coverage_for_gate} errors=${errors})" >&2
  fi

  append_summary_row "$SUMMARY_FILE" "$example_id" "$status" "$rc" "$detected" "$relevant" "$coverage" "$errors" "$policy_fingerprint"
  if [[ "$rc" -ne 0 ]]; then
    overall_rc=1
  fi
done

if [[ "$REQUIRE_UNIQUE_SUMMARY_ROWS" -eq 1 ]]; then
  SUMMARY_CONTRACT_FILE="${OUT_DIR}/summary-contract.tsv"
  if ! evaluate_summary_contract "$SUMMARY_FILE" "$SUMMARY_CONTRACT_FILE"; then
    overall_rc=1
    echo "Summary contract failure: see $SUMMARY_CONTRACT_FILE" >&2
  fi
fi

if [[ "$FAIL_ON_DIFF" -eq 1 ]]; then
  DRIFT_FILE="${OUT_DIR}/drift.tsv"
  if ! evaluate_summary_drift "$BASELINE_FILE" "$SUMMARY_FILE" "$DRIFT_FILE" "$REQUIRE_POLICY_FINGERPRINT_BASELINE" "$REQUIRE_BASELINE_EXAMPLE_PARITY" "$REQUIRE_BASELINE_SCHEMA_VERSION_MATCH" "$BASELINE_SCHEMA_VERSION_FILE" "$SUMMARY_SCHEMA_VERSION_FILE"; then
    overall_rc=1
    echo "Baseline drift failure: see $DRIFT_FILE" >&2
  fi
fi

if [[ -n "$DRIFT_ALLOWLIST_FILE" && -n "$DRIFT_ALLOWLIST_UNUSED_FILE" ]]; then
  unused_allowlist_count="$(write_unused_drift_allowlist_report "$DRIFT_ALLOWLIST_UNUSED_FILE")"
  if [[ "$FAIL_ON_UNUSED_DRIFT_ALLOWLIST" -eq 1 && "$unused_allowlist_count" -gt 0 ]]; then
    overall_rc=1
    echo "Unused drift allowlist entries: $unused_allowlist_count (see $DRIFT_ALLOWLIST_UNUSED_FILE)" >&2
  fi
fi

if [[ "$UPDATE_BASELINE" -eq 1 ]]; then
  mkdir -p "$(dirname "$BASELINE_FILE")"
  cp "$SUMMARY_FILE" "$BASELINE_FILE"
  if [[ -n "$BASELINE_SCHEMA_VERSION_FILE" ]]; then
    mkdir -p "$(dirname "$BASELINE_SCHEMA_VERSION_FILE")"
    cp "$SUMMARY_SCHEMA_VERSION_FILE" "$BASELINE_SCHEMA_VERSION_FILE"
  fi
  echo "Updated baseline: $BASELINE_FILE" >&2
fi

cat "$SUMMARY_FILE"
exit "$overall_rc"

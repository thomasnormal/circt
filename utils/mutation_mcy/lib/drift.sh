#!/usr/bin/env bash
# Shared baseline/schema/drift helpers for mutation MCY runner.

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
summary_schema_contract_fingerprint_from_components() {
  local schema_version="$1"
  local header_line="$2"
  if [[ "$schema_version" == "unknown" || -z "$schema_version" ]]; then
    printf 'unknown\n'
    return 0
  fi
  if [[ -z "$header_line" ]]; then
    printf 'unknown\n'
    return 0
  fi
  hash_string_sha256 "${schema_version}"$'\n'"${header_line}"
}

summary_schema_contract_fingerprint_from_metadata() {
  local fingerprint_raw="$1"
  local fingerprint=""
  fingerprint="$(trim_whitespace "$fingerprint_raw")"
  if [[ "$fingerprint" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' "$fingerprint"
    return 0
  fi
  printf 'unknown\n'
}

summary_schema_contract_fingerprint_for_artifacts() {
  local summary_file="$1"
  local schema_file="${2:-}"
  local contract_file="${3:-}"
  local contract_line=""
  local header_line=""
  local schema_version=""

  if [[ -n "$contract_file" && -f "$contract_file" ]]; then
    if ! IFS= read -r contract_line < "$contract_file"; then
      printf 'unknown\n'
      return 0
    fi
    summary_schema_contract_fingerprint_from_metadata "$contract_line"
    return 0
  fi

  if ! IFS= read -r header_line < "$summary_file"; then
    printf 'unknown\n'
    return 0
  fi
  schema_version="$(summary_schema_version_for_artifacts "$summary_file" "$schema_file")"
  summary_schema_contract_fingerprint_from_components "$schema_version" "$header_line"
}


retry_reason_schema_version_from_header() {
  local header_line="$1"
  if [[ "$header_line" == "$RETRY_REASON_SUMMARY_HEADER_V1" ]]; then
    printf 'v1\n'
    return 0
  fi
  printf 'unknown\n'
}

retry_reason_schema_version_for_artifacts() {
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
  retry_reason_schema_version_from_header "$header_line"
}

retry_reason_schema_contract_fingerprint_for_artifacts() {
  local summary_file="$1"
  local schema_file="${2:-}"
  local contract_file="${3:-}"
  local contract_line=""
  local header_line=""
  local schema_version=""

  if [[ -n "$contract_file" && -f "$contract_file" ]]; then
    if ! IFS= read -r contract_line < "$contract_file"; then
      printf 'unknown\n'
      return 0
    fi
    summary_schema_contract_fingerprint_from_metadata "$contract_line"
    return 0
  fi

  if ! IFS= read -r header_line < "$summary_file"; then
    printf 'unknown\n'
    return 0
  fi
  schema_version="$(retry_reason_schema_version_for_artifacts "$summary_file" "$schema_file")"
  summary_schema_contract_fingerprint_from_components "$schema_version" "$header_line"
}

migrate_baseline_schema_artifacts() {
  local baseline_file="$1"
  local baseline_schema_version_file="$2"
  local baseline_schema_contract_file="$3"
  local header_line=""
  local schema_version=""
  local schema_contract=""

  if ! IFS= read -r header_line < "$baseline_file"; then
    echo "Baseline file is empty: $baseline_file" >&2
    return 1
  fi

  schema_version="$(summary_schema_version_from_header "$header_line")"
  if [[ "$schema_version" == "unknown" ]]; then
    echo "Unable to infer baseline schema version from header in $baseline_file" >&2
    return 1
  fi

  schema_contract="$(summary_schema_contract_fingerprint_from_components "$schema_version" "$header_line")"
  if [[ "$schema_contract" == "unknown" ]]; then
    echo "Unable to infer baseline schema contract from header in $baseline_file" >&2
    return 1
  fi

  mkdir -p "$(dirname "$baseline_schema_version_file")"
  printf '%s\n' "$schema_version" > "$baseline_schema_version_file"
  mkdir -p "$(dirname "$baseline_schema_contract_file")"
  printf '%s\n' "$schema_contract" > "$baseline_schema_contract_file"

  echo "Migrated baseline schema artifacts: $baseline_schema_version_file $baseline_schema_contract_file" >&2
  return 0
}

migrate_retry_reason_baseline_schema_artifacts() {
  local baseline_file="$1"
  local baseline_schema_version_file="$2"
  local baseline_schema_contract_file="$3"
  local header_line=""
  local schema_version=""
  local schema_contract=""

  if ! IFS= read -r header_line < "$baseline_file"; then
    echo "Retry-reason baseline file is empty: $baseline_file" >&2
    return 1
  fi

  schema_version="$(retry_reason_schema_version_from_header "$header_line")"
  if [[ "$schema_version" == "unknown" ]]; then
    echo "Unable to infer retry-reason baseline schema version from header in $baseline_file" >&2
    return 1
  fi

  schema_contract="$(summary_schema_contract_fingerprint_from_components "$schema_version" "$header_line")"
  if [[ "$schema_contract" == "unknown" ]]; then
    echo "Unable to infer retry-reason baseline schema contract from header in $baseline_file" >&2
    return 1
  fi

  mkdir -p "$(dirname "$baseline_schema_version_file")"
  printf '%s\n' "$schema_version" > "$baseline_schema_version_file"
  mkdir -p "$(dirname "$baseline_schema_contract_file")"
  printf '%s\n' "$schema_contract" > "$baseline_schema_contract_file"

  echo "Migrated retry-reason baseline schema artifacts: $baseline_schema_version_file $baseline_schema_contract_file" >&2
  return 0
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
  local require_baseline_schema_contract_match="${9:-0}"
  local baseline_schema_contract_file="${10:-}"
  local summary_schema_contract_file="${11:-}"
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
  local baseline_total_detected=0
  local baseline_total_relevant=0
  local baseline_total_errors=0
  local summary_total_detected=0
  local summary_total_relevant=0
  local summary_total_errors=0
  local baseline_total_coverage="0.00"
  local summary_total_coverage="0.00"

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

  local baseline_schema_contract=""
  local summary_schema_contract=""
  local max_detected_drop="$MAX_DETECTED_DROP"
  local max_detected_drop_percent="$MAX_DETECTED_DROP_PERCENT"
  local max_relevant_drop="$MAX_RELEVANT_DROP"
  local max_relevant_drop_percent="$MAX_RELEVANT_DROP_PERCENT"
  local max_coverage_drop_percent="$MAX_COVERAGE_DROP_PERCENT"
  local max_errors_increase="$MAX_ERRORS_INCREASE"
  local max_total_detected_drop="$MAX_TOTAL_DETECTED_DROP"
  local max_total_detected_drop_percent="$MAX_TOTAL_DETECTED_DROP_PERCENT"
  local max_total_relevant_drop="$MAX_TOTAL_RELEVANT_DROP"
  local max_total_relevant_drop_percent="$MAX_TOTAL_RELEVANT_DROP_PERCENT"
  local max_total_coverage_drop_percent="$MAX_TOTAL_COVERAGE_DROP_PERCENT"
  local max_total_errors_increase="$MAX_TOTAL_ERRORS_INCREASE"
  summary_schema_contract="$(summary_schema_contract_fingerprint_for_artifacts "$summary_file" "$summary_schema_version_file" "$summary_schema_contract_file")"
  baseline_schema_contract="$(summary_schema_contract_fingerprint_for_artifacts "$baseline_file" "$baseline_schema_version_file" "$baseline_schema_contract_file")"

  if [[ "$require_baseline_schema_contract_match" -eq 1 ]]; then
    if [[ "$baseline_schema_contract" != "$summary_schema_contract" ]]; then
      if ! append_drift_candidate "$drift_file" "__baseline__" "baseline_schema_contract" "$baseline_schema_contract" "$summary_schema_contract" "baseline_schema_contract_mismatch"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "__baseline__" "baseline_schema_contract" "$baseline_schema_contract" "$summary_schema_contract" "ok" ""
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

    baseline_total_detected=$((baseline_total_detected + $(normalize_int_or_zero "${_detected:-0}")))
    baseline_total_relevant=$((baseline_total_relevant + $(normalize_int_or_zero "${_relevant:-0}")))
    baseline_total_errors=$((baseline_total_errors + $(normalize_int_or_zero "${_errors:-0}")))
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
    summary_total_detected=$((summary_total_detected + detected))
    summary_total_relevant=$((summary_total_relevant + relevant))
    summary_total_errors=$((summary_total_errors + errors))
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

    if [[ ${#EXAMPLE_BASELINE_HISTORY_FILES[@]} -gt 0 ]]; then
      compute_example_metric_percentile_from_histories "$example" 4 "$_bd" int || return 1
      _bd="$COMPUTED_EXAMPLE_HISTORY_METRIC"
      compute_example_metric_percentile_from_histories "$example" 5 "$_br" int || return 1
      _br="$COMPUTED_EXAMPLE_HISTORY_METRIC"
      compute_example_metric_percentile_from_histories "$example" 6 "$base_cov_num" decimal || return 1
      base_cov_num="$COMPUTED_EXAMPLE_HISTORY_METRIC"
      compute_example_metric_percentile_from_histories "$example" 7 "$_berr" int || return 1
      _berr="$COMPUTED_EXAMPLE_HISTORY_METRIC"
    fi

    if [[ "${_bs:-}" == "PASS" && "$status" != "PASS" ]]; then
      if ! append_drift_candidate "$drift_file" "$example" "status" "${_bs:-}" "$status" "status_regressed"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "$example" "status" "${_bs:-}" "$status" "ok" ""
    fi

    local effective_max_detected_drop="$max_detected_drop"
    local effective_max_detected_drop_percent="$max_detected_drop_percent"
    local effective_max_relevant_drop="$max_relevant_drop"
    local effective_max_relevant_drop_percent="$max_relevant_drop_percent"
    local effective_max_coverage_drop_percent="$max_coverage_drop_percent"
    local effective_max_errors_increase="$max_errors_increase"

    if [[ -n "${EXAMPLE_TO_MAX_DETECTED_DROP[$example]+x}" ]]; then
      effective_max_detected_drop="${EXAMPLE_TO_MAX_DETECTED_DROP[$example]}"
    fi
    if [[ -n "${EXAMPLE_TO_MAX_RELEVANT_DROP[$example]+x}" ]]; then
      effective_max_relevant_drop="${EXAMPLE_TO_MAX_RELEVANT_DROP[$example]}"
    fi
    if [[ -n "${EXAMPLE_TO_MAX_DETECTED_DROP_PERCENT[$example]+x}" ]]; then
      effective_max_detected_drop_percent="${EXAMPLE_TO_MAX_DETECTED_DROP_PERCENT[$example]}"
    fi
    if [[ -n "${EXAMPLE_TO_MAX_RELEVANT_DROP_PERCENT[$example]+x}" ]]; then
      effective_max_relevant_drop_percent="${EXAMPLE_TO_MAX_RELEVANT_DROP_PERCENT[$example]}"
    fi
    if [[ -n "${EXAMPLE_TO_MAX_COVERAGE_DROP_PERCENT[$example]+x}" ]]; then
      effective_max_coverage_drop_percent="${EXAMPLE_TO_MAX_COVERAGE_DROP_PERCENT[$example]}"
    fi
    if [[ -n "${EXAMPLE_TO_MAX_ERRORS_INCREASE[$example]+x}" ]]; then
      effective_max_errors_increase="${EXAMPLE_TO_MAX_ERRORS_INCREASE[$example]}"
    fi

    local detected_drop_percent_allowance=0
    local relevant_drop_percent_allowance=0
    detected_drop_percent_allowance="$(percentage_ceiling_count "$_bd" "$effective_max_detected_drop_percent")"
    relevant_drop_percent_allowance="$(percentage_ceiling_count "$_br" "$effective_max_relevant_drop_percent")"

    if (( detected + effective_max_detected_drop + detected_drop_percent_allowance < _bd )); then
      if ! append_drift_candidate "$drift_file" "$example" "detected_mutants" "$_bd" "$detected" "detected_decreased"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "$example" "detected_mutants" "$_bd" "$detected" "ok" ""
    fi

    local per_example_min_coverage
    per_example_min_coverage="$(awk -v b="$base_cov_num" -v d="$effective_max_coverage_drop_percent" 'BEGIN { v=b-d; if (v < 0) v=0; printf "%.6f", v }')"
    if float_lt "$coverage_num" "$per_example_min_coverage"; then
      if ! append_drift_candidate "$drift_file" "$example" "coverage_percent" "$base_cov_num" "$coverage_num" "coverage_decreased"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "$example" "coverage_percent" "$base_cov_num" "$coverage_num" "ok" ""
    fi

    if (( errors > _berr + effective_max_errors_increase )); then
      if ! append_drift_candidate "$drift_file" "$example" "errors" "$_berr" "$errors" "errors_increased"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "$example" "errors" "$_berr" "$errors" "ok" ""
    fi

    if (( relevant + effective_max_relevant_drop + relevant_drop_percent_allowance < _br )); then
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

  if [[ "$baseline_total_relevant" -gt 0 ]]; then
    baseline_total_coverage="$(awk -v d="$baseline_total_detected" -v r="$baseline_total_relevant" 'BEGIN { printf "%.2f", (100.0 * d) / r }')"
  fi
  if [[ "$summary_total_relevant" -gt 0 ]]; then
    summary_total_coverage="$(awk -v d="$summary_total_detected" -v r="$summary_total_relevant" 'BEGIN { printf "%.2f", (100.0 * d) / r }')"
  fi

  if [[ ${#SUITE_BASELINE_HISTORY_FILES[@]} -gt 0 ]]; then
    local history_detected_file history_relevant_file history_coverage_file history_errors_file
    local history_file history_detected history_relevant history_coverage history_errors
    history_detected_file="$(mktemp "${OUT_DIR}/suite-history-detected.XXXXXX")"
    history_relevant_file="$(mktemp "${OUT_DIR}/suite-history-relevant.XXXXXX")"
    history_coverage_file="$(mktemp "${OUT_DIR}/suite-history-coverage.XXXXXX")"
    history_errors_file="$(mktemp "${OUT_DIR}/suite-history-errors.XXXXXX")"
    printf '%s\n' "$baseline_total_detected" > "$history_detected_file"
    printf '%s\n' "$baseline_total_relevant" > "$history_relevant_file"
    printf '%s\n' "$baseline_total_coverage" > "$history_coverage_file"
    printf '%s\n' "$baseline_total_errors" > "$history_errors_file"
    for history_file in "${SUITE_BASELINE_HISTORY_FILES[@]}"; do
      IFS=$'	' read -r history_detected history_relevant history_coverage history_errors <<< "$(compute_suite_totals_from_summary "$history_file")"
      printf '%s\n' "$history_detected" >> "$history_detected_file"
      printf '%s\n' "$history_relevant" >> "$history_relevant_file"
      printf '%s\n' "$history_coverage" >> "$history_coverage_file"
      printf '%s\n' "$history_errors" >> "$history_errors_file"
    done
    local effective_detected_mode effective_relevant_mode effective_coverage_mode effective_errors_mode
    effective_detected_mode="$(resolved_history_metric_aggregation_mode detected)"
    effective_relevant_mode="$(resolved_history_metric_aggregation_mode relevant)"
    effective_coverage_mode="$(resolved_history_metric_aggregation_mode coverage)"
    effective_errors_mode="$(resolved_history_metric_aggregation_mode errors)"

    if [[ "$effective_detected_mode" == "ewma" ]]; then
      baseline_total_detected="$(ewma_int_from_file "$history_detected_file" "$HISTORY_EWMA_ALPHA")"
    elif [[ "$effective_detected_mode" == "max" ]]; then
      baseline_total_detected="$(max_int_from_file "$history_detected_file")"
    else
      baseline_total_detected="$(percentile_int_from_file "$history_detected_file" "$SUITE_HISTORY_PERCENTILE")"
    fi

    if [[ "$effective_relevant_mode" == "ewma" ]]; then
      baseline_total_relevant="$(ewma_int_from_file "$history_relevant_file" "$HISTORY_EWMA_ALPHA")"
    elif [[ "$effective_relevant_mode" == "max" ]]; then
      baseline_total_relevant="$(max_int_from_file "$history_relevant_file")"
    else
      baseline_total_relevant="$(percentile_int_from_file "$history_relevant_file" "$SUITE_HISTORY_PERCENTILE")"
    fi

    if [[ "$effective_coverage_mode" == "ewma" ]]; then
      baseline_total_coverage="$(ewma_decimal_from_file "$history_coverage_file" "$HISTORY_EWMA_ALPHA")"
    elif [[ "$effective_coverage_mode" == "max" ]]; then
      baseline_total_coverage="$(max_decimal_from_file "$history_coverage_file")"
    else
      baseline_total_coverage="$(percentile_decimal_from_file "$history_coverage_file" "$SUITE_HISTORY_PERCENTILE")"
    fi

    if [[ "$effective_errors_mode" == "ewma" ]]; then
      baseline_total_errors="$(ewma_int_from_file "$history_errors_file" "$HISTORY_EWMA_ALPHA")"
    elif [[ "$effective_errors_mode" == "max" ]]; then
      baseline_total_errors="$(max_int_from_file "$history_errors_file")"
    else
      baseline_total_errors="$(percentile_int_from_file "$history_errors_file" "$SUITE_HISTORY_PERCENTILE")"
    fi
    rm -f "$history_detected_file" "$history_relevant_file" "$history_coverage_file" "$history_errors_file"
  fi

  local suite_detected_drop_percent_allowance=0
  local suite_relevant_drop_percent_allowance=0
  suite_detected_drop_percent_allowance="$(percentage_ceiling_count "$baseline_total_detected" "$max_total_detected_drop_percent")"
  suite_relevant_drop_percent_allowance="$(percentage_ceiling_count "$baseline_total_relevant" "$max_total_relevant_drop_percent")"

  if (( summary_total_detected + max_total_detected_drop + suite_detected_drop_percent_allowance < baseline_total_detected )); then
    if ! append_drift_candidate "$drift_file" "__suite__" "suite_detected_mutants" "$baseline_total_detected" "$summary_total_detected" "detected_decreased"; then
      regressions=$((regressions + 1))
    fi
  else
    append_drift_row "$drift_file" "__suite__" "suite_detected_mutants" "$baseline_total_detected" "$summary_total_detected" "ok" ""
  fi

  if (( summary_total_relevant + max_total_relevant_drop + suite_relevant_drop_percent_allowance < baseline_total_relevant )); then
    if ! append_drift_candidate "$drift_file" "__suite__" "suite_relevant_mutants" "$baseline_total_relevant" "$summary_total_relevant" "relevant_decreased"; then
      regressions=$((regressions + 1))
    fi
  else
    append_drift_row "$drift_file" "__suite__" "suite_relevant_mutants" "$baseline_total_relevant" "$summary_total_relevant" "ok" ""
  fi

  local suite_min_coverage
  suite_min_coverage="$(awk -v b="$baseline_total_coverage" -v d="$max_total_coverage_drop_percent" 'BEGIN { v=b-d; if (v < 0) v=0; printf "%.6f", v }')"
  if float_lt "$summary_total_coverage" "$suite_min_coverage"; then
    if ! append_drift_candidate "$drift_file" "__suite__" "suite_coverage_percent" "$baseline_total_coverage" "$summary_total_coverage" "coverage_decreased"; then
      regressions=$((regressions + 1))
    fi
  else
    append_drift_row "$drift_file" "__suite__" "suite_coverage_percent" "$baseline_total_coverage" "$summary_total_coverage" "ok" ""
  fi

  if (( summary_total_errors > baseline_total_errors + max_total_errors_increase )); then
    if ! append_drift_candidate "$drift_file" "__suite__" "suite_errors" "$baseline_total_errors" "$summary_total_errors" "errors_increased"; then
      regressions=$((regressions + 1))
    fi
  else
    append_drift_row "$drift_file" "__suite__" "suite_errors" "$baseline_total_errors" "$summary_total_errors" "ok" ""
  fi

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

evaluate_retry_reason_drift() {
  local baseline_file="$1"
  local summary_file="$2"
  local drift_file="$3"
  local baseline_schema_version_file="${4:-}"
  local summary_schema_version_file="${5:-}"
  local baseline_schema_contract_file="${6:-}"
  local summary_schema_contract_file="${7:-}"
  local regressions=0
  local baseline_reason=""
  local baseline_retries=""
  local current_reason=""
  local current_retries=""
  local reason=""
  local absolute_tolerance=0
  local percent_tolerance="0"
  local percent_tolerance_count=0
  local suite_absolute_tolerance="${RETRY_REASON_DRIFT_SUITE_TOLERANCE}"
  local suite_percent_tolerance="${RETRY_REASON_DRIFT_SUITE_PERCENT_TOLERANCE}"
  local suite_percent_tolerance_count=0
  local baseline_total_retries=0
  local current_total_retries=0
  local suite_allowed_retries=0
  local per_reason_allowed_retries=0
  local -A baseline_counts=()
  local -A current_counts=()
  local -A baseline_duplicate_seen=()
  local -A current_duplicate_seen=()
  local -a baseline_duplicates=()
  local -a current_duplicates=()
  local baseline_schema_version=""
  local summary_schema_version=""
  local baseline_schema_contract=""
  local summary_schema_contract=""

  printf 'example\tmetric\tbaseline\tcurrent\toutcome\tdetail\n' > "$drift_file"

  summary_schema_version="$(retry_reason_schema_version_for_artifacts "$summary_file" "$summary_schema_version_file")"
  baseline_schema_version="$(retry_reason_schema_version_for_artifacts "$baseline_file" "$baseline_schema_version_file")"
  if [[ "$REQUIRE_RETRY_REASON_SCHEMA_ARTIFACT_VALIDITY" -eq 1 ]]; then
    if [[ "$baseline_schema_version" == "unknown" ]]; then
      if ! append_drift_candidate "$drift_file" "__baseline__" "retry_reason_schema_version" "known" "$baseline_schema_version" "retry_reason_schema_version_unknown_baseline"; then
        regressions=$((regressions + 1))
      fi
    fi
    if [[ "$summary_schema_version" == "unknown" ]]; then
      if ! append_drift_candidate "$drift_file" "__baseline__" "retry_reason_schema_version" "known" "$summary_schema_version" "retry_reason_schema_version_unknown_current"; then
        regressions=$((regressions + 1))
      fi
    fi
  fi
  if [[ "$baseline_schema_version" != "$summary_schema_version" ]]; then
    if ! append_drift_candidate "$drift_file" "__baseline__" "retry_reason_schema_version" "$baseline_schema_version" "$summary_schema_version" "retry_reason_schema_version_mismatch"; then
      regressions=$((regressions + 1))
    fi
  else
    append_drift_row "$drift_file" "__baseline__" "retry_reason_schema_version" "$baseline_schema_version" "$summary_schema_version" "ok" ""
  fi

  summary_schema_contract="$(retry_reason_schema_contract_fingerprint_for_artifacts "$summary_file" "$summary_schema_version_file" "$summary_schema_contract_file")"
  baseline_schema_contract="$(retry_reason_schema_contract_fingerprint_for_artifacts "$baseline_file" "$baseline_schema_version_file" "$baseline_schema_contract_file")"
  if [[ "$REQUIRE_RETRY_REASON_SCHEMA_ARTIFACT_VALIDITY" -eq 1 ]]; then
    if [[ "$baseline_schema_contract" == "unknown" ]]; then
      if ! append_drift_candidate "$drift_file" "__baseline__" "retry_reason_schema_contract" "known" "$baseline_schema_contract" "retry_reason_schema_contract_unknown_baseline"; then
        regressions=$((regressions + 1))
      fi
    fi
    if [[ "$summary_schema_contract" == "unknown" ]]; then
      if ! append_drift_candidate "$drift_file" "__baseline__" "retry_reason_schema_contract" "known" "$summary_schema_contract" "retry_reason_schema_contract_unknown_current"; then
        regressions=$((regressions + 1))
      fi
    fi
  fi
  if [[ "$baseline_schema_contract" != "$summary_schema_contract" ]]; then
    if ! append_drift_candidate "$drift_file" "__baseline__" "retry_reason_schema_contract" "$baseline_schema_contract" "$summary_schema_contract" "retry_reason_schema_contract_mismatch"; then
      regressions=$((regressions + 1))
    fi
  else
    append_drift_row "$drift_file" "__baseline__" "retry_reason_schema_contract" "$baseline_schema_contract" "$summary_schema_contract" "ok" ""
  fi

  while IFS=$'\t' read -r baseline_reason baseline_retries; do
    [[ "$baseline_reason" == "retry_reason" ]] && continue
    baseline_reason="$(trim_whitespace "${baseline_reason:-}")"
    if [[ -z "$baseline_reason" ]]; then
      continue
    fi
    baseline_retries="$(normalize_int_or_zero "${baseline_retries:-0}")"
    if [[ -n "${baseline_counts[$baseline_reason]+x}" ]]; then
      if [[ -z "${baseline_duplicate_seen[$baseline_reason]+x}" ]]; then
        baseline_duplicate_seen["$baseline_reason"]=1
        baseline_duplicates+=("$baseline_reason")
      fi
      continue
    fi
    baseline_counts["$baseline_reason"]="$baseline_retries"
    baseline_total_retries=$((baseline_total_retries + baseline_retries))
  done < "$baseline_file"

  while IFS=$'\t' read -r current_reason current_retries; do
    [[ "$current_reason" == "retry_reason" ]] && continue
    current_reason="$(trim_whitespace "${current_reason:-}")"
    if [[ -z "$current_reason" ]]; then
      continue
    fi
    current_retries="$(normalize_int_or_zero "${current_retries:-0}")"
    if [[ -n "${current_counts[$current_reason]+x}" ]]; then
      if [[ -z "${current_duplicate_seen[$current_reason]+x}" ]]; then
        current_duplicate_seen["$current_reason"]=1
        current_duplicates+=("$current_reason")
      fi
      continue
    fi
    current_counts["$current_reason"]="$current_retries"
    current_total_retries=$((current_total_retries + current_retries))
  done < "$summary_file"

  for reason in "${!baseline_counts[@]}"; do
    baseline_retries="${baseline_counts[$reason]}"
    current_retries="${current_counts[$reason]:-0}"
    absolute_tolerance="${RETRY_REASON_DRIFT_TOLERANCE_MAP[$reason]:-0}"
    percent_tolerance="${RETRY_REASON_DRIFT_PERCENT_TOLERANCE_MAP[$reason]:-0}"
    percent_tolerance_count="$(percentage_ceiling_count "$baseline_retries" "$percent_tolerance")"
    per_reason_allowed_retries=$((baseline_retries + absolute_tolerance + percent_tolerance_count))
    if [[ "$current_retries" -gt "$per_reason_allowed_retries" ]]; then
      if ! append_drift_candidate "$drift_file" "$reason" "retry_count" "$baseline_retries" "$current_retries" "retry_count_increased_over_tolerance"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "$reason" "retry_count" "$baseline_retries" "$current_retries" "ok" ""
    fi
  done

  if [[ "$REQUIRE_RETRY_REASON_BASELINE_PARITY" -eq 1 ]]; then
    for reason in "${!baseline_counts[@]}"; do
      if [[ -z "${current_counts[$reason]+x}" ]]; then
        if ! append_drift_candidate "$drift_file" "$reason" "row" "present" "missing" "missing_current_row"; then
          regressions=$((regressions + 1))
        fi
      fi
    done
  fi

  for reason in "${!current_counts[@]}"; do
    if [[ -n "${baseline_counts[$reason]+x}" ]]; then
      continue
    fi
    if [[ "$REQUIRE_RETRY_REASON_BASELINE_PARITY" -eq 1 ]]; then
      if ! append_drift_candidate "$drift_file" "$reason" "row" "absent" "present" "unexpected_current_row"; then
        regressions=$((regressions + 1))
      fi
      continue
    fi
    current_retries="${current_counts[$reason]}"
    absolute_tolerance="${RETRY_REASON_DRIFT_TOLERANCE_MAP[$reason]:-0}"
    if [[ "$current_retries" -gt "$absolute_tolerance" ]]; then
      if ! append_drift_candidate "$drift_file" "$reason" "retry_count" "0" "$current_retries" "new_retry_reason_over_tolerance"; then
        regressions=$((regressions + 1))
      fi
    else
      append_drift_row "$drift_file" "$reason" "retry_count" "0" "$current_retries" "ok" ""
    fi
  done

  suite_percent_tolerance_count="$(percentage_ceiling_count "$baseline_total_retries" "$suite_percent_tolerance")"
  suite_allowed_retries=$((baseline_total_retries + suite_absolute_tolerance + suite_percent_tolerance_count))
  if [[ "$current_total_retries" -gt "$suite_allowed_retries" ]]; then
    if ! append_drift_candidate "$drift_file" "__suite__" "suite_retries" "$baseline_total_retries" "$current_total_retries" "retries_increased_over_tolerance"; then
      regressions=$((regressions + 1))
    fi
  else
    append_drift_row "$drift_file" "__suite__" "suite_retries" "$baseline_total_retries" "$current_total_retries" "ok" ""
  fi

  for reason in "${baseline_duplicates[@]}"; do
    if ! append_drift_candidate "$drift_file" "$reason" "row" "single_row" "duplicate_rows" "duplicate_baseline_row"; then
      regressions=$((regressions + 1))
    fi
  done

  for reason in "${current_duplicates[@]}"; do
    if ! append_drift_candidate "$drift_file" "$reason" "row" "single_row" "duplicate_rows" "duplicate_current_row"; then
      regressions=$((regressions + 1))
    fi
  done

  if [[ "$regressions" -gt 0 ]]; then
    return 1
  fi
  return 0
}

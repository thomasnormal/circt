#!/usr/bin/env bash
# Shared manifest parsing for mutation MCY runner.

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
  local example_timeout_override=""
  local example_retries_override=""
  local example_retry_delay_ms_override=""
  local max_detected_drop_override=""
  local max_relevant_drop_override=""
  local max_coverage_drop_percent_override=""
  local max_errors_increase_override=""
  local max_detected_drop_percent_override=""
  local max_relevant_drop_percent_override=""
  local max_total_detected_drop_override=""
  local max_total_detected_drop_percent_override=""
  local max_total_relevant_drop_override=""
  local max_total_relevant_drop_percent_override=""
  local max_total_coverage_drop_percent_override=""
  local max_total_errors_increase_override=""
  local native_real_harness_override=""
  local native_mutation_ops_override=""
  local native_real_harness_args_override=""
  local extra=""
  local resolved_design=""
  local manifest_max_total_detected_drop_override=""
  local manifest_max_total_detected_drop_percent_override=""
  local manifest_max_total_relevant_drop_override=""
  local manifest_max_total_relevant_drop_percent_override=""
  local manifest_max_total_coverage_drop_percent_override=""
  local manifest_max_total_errors_increase_override=""

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
      mutation_limit_override example_timeout_override example_retries_override \
      example_retry_delay_ms_override max_detected_drop_override \
      max_relevant_drop_override max_coverage_drop_percent_override \
      max_errors_increase_override max_detected_drop_percent_override \
      max_relevant_drop_percent_override max_total_detected_drop_override \
      max_total_detected_drop_percent_override max_total_relevant_drop_override \
      max_total_relevant_drop_percent_override \
      max_total_coverage_drop_percent_override max_total_errors_increase_override \
      native_real_harness_override native_mutation_ops_override \
      native_real_harness_args_override extra <<< "$line"

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
    example_timeout_override="$(normalize_manifest_optional "${example_timeout_override:-}")"
    example_retries_override="$(normalize_manifest_optional "${example_retries_override:-}")"
    example_retry_delay_ms_override="$(normalize_manifest_optional "${example_retry_delay_ms_override:-}")"
    max_detected_drop_override="$(normalize_manifest_optional "${max_detected_drop_override:-}")"
    max_relevant_drop_override="$(normalize_manifest_optional "${max_relevant_drop_override:-}")"
    max_coverage_drop_percent_override="$(normalize_manifest_optional "${max_coverage_drop_percent_override:-}")"
    max_errors_increase_override="$(normalize_manifest_optional "${max_errors_increase_override:-}")"
    max_detected_drop_percent_override="$(normalize_manifest_optional "${max_detected_drop_percent_override:-}")"
    max_relevant_drop_percent_override="$(normalize_manifest_optional "${max_relevant_drop_percent_override:-}")"
    max_total_detected_drop_override="$(normalize_manifest_optional "${max_total_detected_drop_override:-}")"
    max_total_detected_drop_percent_override="$(normalize_manifest_optional "${max_total_detected_drop_percent_override:-}")"
    max_total_relevant_drop_override="$(normalize_manifest_optional "${max_total_relevant_drop_override:-}")"
    max_total_relevant_drop_percent_override="$(normalize_manifest_optional "${max_total_relevant_drop_percent_override:-}")"
    max_total_coverage_drop_percent_override="$(normalize_manifest_optional "${max_total_coverage_drop_percent_override:-}")"
    max_total_errors_increase_override="$(normalize_manifest_optional "${max_total_errors_increase_override:-}")"
    native_real_harness_override="$(normalize_manifest_optional "${native_real_harness_override:-}")"
    if [[ -n "$native_real_harness_override" && "$native_real_harness_override" != /* ]]; then
      native_real_harness_override="${EXAMPLES_ROOT}/${native_real_harness_override}"
    fi
    if [[ -n "$native_real_harness_override" ]]; then
      native_real_harness_override="$(canonicalize_path_for_policy "$native_real_harness_override")"
    fi
    native_mutation_ops_override="$(normalize_manifest_optional "${native_mutation_ops_override:-}")"
    native_real_harness_args_override="$(normalize_manifest_optional "${native_real_harness_args_override:-}")"
    extra="$(trim_whitespace "${extra:-}")"

    if [[ -z "$example_id" || -z "$design" || -z "$top" || -n "$extra" ]]; then
      echo "Invalid example manifest row ${line_no} in ${file} (expected: example<TAB>design<TAB>top with up to 27 optional override columns)." >&2
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
    if [[ -n "$example_timeout_override" && ! "$example_timeout_override" =~ ^[0-9]+$ ]]; then
      echo "Invalid example_timeout_sec override in manifest row ${line_no}: ${example_timeout_override}" >&2
      return 1
    fi
    if [[ -n "$example_retries_override" && ! "$example_retries_override" =~ ^[0-9]+$ ]]; then
      echo "Invalid example_retries override in manifest row ${line_no}: ${example_retries_override}" >&2
      return 1
    fi
    if [[ -n "$example_retry_delay_ms_override" && ! "$example_retry_delay_ms_override" =~ ^[0-9]+$ ]]; then
      echo "Invalid example_retry_delay_ms override in manifest row ${line_no}: ${example_retry_delay_ms_override}" >&2
      return 1
    fi
    if [[ -n "$max_detected_drop_override" && ! "$max_detected_drop_override" =~ ^[0-9]+$ ]]; then
      echo "Invalid max_detected_drop override in manifest row ${line_no}: ${max_detected_drop_override}" >&2
      return 1
    fi
    if [[ -n "$max_relevant_drop_override" && ! "$max_relevant_drop_override" =~ ^[0-9]+$ ]]; then
      echo "Invalid max_relevant_drop override in manifest row ${line_no}: ${max_relevant_drop_override}" >&2
      return 1
    fi
    if [[ -n "$max_coverage_drop_percent_override" && ! "$max_coverage_drop_percent_override" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "Invalid max_coverage_drop_percent override in manifest row ${line_no}: ${max_coverage_drop_percent_override}" >&2
      return 1
    fi
    if [[ -n "$max_coverage_drop_percent_override" ]] && ! awk -v v="$max_coverage_drop_percent_override" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
      echo "Invalid max_coverage_drop_percent override in manifest row ${line_no}: ${max_coverage_drop_percent_override}" >&2
      return 1
    fi
    if [[ -n "$max_errors_increase_override" && ! "$max_errors_increase_override" =~ ^[0-9]+$ ]]; then
      echo "Invalid max_errors_increase override in manifest row ${line_no}: ${max_errors_increase_override}" >&2
      return 1
    fi
    if [[ -n "$max_detected_drop_percent_override" && ! "$max_detected_drop_percent_override" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "Invalid max_detected_drop_percent override in manifest row ${line_no}: ${max_detected_drop_percent_override}" >&2
      return 1
    fi
    if [[ -n "$max_detected_drop_percent_override" ]] && ! awk -v v="$max_detected_drop_percent_override" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
      echo "Invalid max_detected_drop_percent override in manifest row ${line_no}: ${max_detected_drop_percent_override}" >&2
      return 1
    fi
    if [[ -n "$max_relevant_drop_percent_override" && ! "$max_relevant_drop_percent_override" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "Invalid max_relevant_drop_percent override in manifest row ${line_no}: ${max_relevant_drop_percent_override}" >&2
      return 1
    fi
    if [[ -n "$max_relevant_drop_percent_override" ]] && ! awk -v v="$max_relevant_drop_percent_override" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
      echo "Invalid max_relevant_drop_percent override in manifest row ${line_no}: ${max_relevant_drop_percent_override}" >&2
      return 1
    fi
    if [[ -n "$max_total_detected_drop_override" && ! "$max_total_detected_drop_override" =~ ^[0-9]+$ ]]; then
      echo "Invalid max_total_detected_drop override in manifest row ${line_no}: ${max_total_detected_drop_override}" >&2
      return 1
    fi
    if [[ -n "$max_total_detected_drop_percent_override" && ! "$max_total_detected_drop_percent_override" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "Invalid max_total_detected_drop_percent override in manifest row ${line_no}: ${max_total_detected_drop_percent_override}" >&2
      return 1
    fi
    if [[ -n "$max_total_detected_drop_percent_override" ]] && ! awk -v v="$max_total_detected_drop_percent_override" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
      echo "Invalid max_total_detected_drop_percent override in manifest row ${line_no}: ${max_total_detected_drop_percent_override}" >&2
      return 1
    fi
    if [[ -n "$max_total_relevant_drop_override" && ! "$max_total_relevant_drop_override" =~ ^[0-9]+$ ]]; then
      echo "Invalid max_total_relevant_drop override in manifest row ${line_no}: ${max_total_relevant_drop_override}" >&2
      return 1
    fi
    if [[ -n "$max_total_relevant_drop_percent_override" && ! "$max_total_relevant_drop_percent_override" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "Invalid max_total_relevant_drop_percent override in manifest row ${line_no}: ${max_total_relevant_drop_percent_override}" >&2
      return 1
    fi
    if [[ -n "$max_total_relevant_drop_percent_override" ]] && ! awk -v v="$max_total_relevant_drop_percent_override" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
      echo "Invalid max_total_relevant_drop_percent override in manifest row ${line_no}: ${max_total_relevant_drop_percent_override}" >&2
      return 1
    fi
    if [[ -n "$max_total_coverage_drop_percent_override" && ! "$max_total_coverage_drop_percent_override" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "Invalid max_total_coverage_drop_percent override in manifest row ${line_no}: ${max_total_coverage_drop_percent_override}" >&2
      return 1
    fi
    if [[ -n "$max_total_coverage_drop_percent_override" ]] && ! awk -v v="$max_total_coverage_drop_percent_override" 'BEGIN { exit !(v >= 0 && v <= 100) }'; then
      echo "Invalid max_total_coverage_drop_percent override in manifest row ${line_no}: ${max_total_coverage_drop_percent_override}" >&2
      return 1
    fi
    if [[ -n "$max_total_errors_increase_override" && ! "$max_total_errors_increase_override" =~ ^[0-9]+$ ]]; then
      echo "Invalid max_total_errors_increase override in manifest row ${line_no}: ${max_total_errors_increase_override}" >&2
      return 1
    fi
    if [[ -n "$mutations_mode_counts_override" && -n "$mutations_mode_weights_override" ]]; then
      echo "Manifest row ${line_no} sets both mutations_mode_counts and mutations_mode_weights; choose one." >&2
      return 1
    fi
    if [[ -n "$native_mutation_ops_override" ]]; then
      if ! validate_native_mutation_ops_spec "$native_mutation_ops_override" "manifest row ${line_no}: invalid native_mutation_ops override"; then
        return 1
      fi
      native_mutation_ops_override="$(canonicalize_native_mutation_ops_spec "$native_mutation_ops_override")"
    fi
    if [[ -n "$native_real_harness_args_override" ]]; then
      if ! validate_native_real_harness_args_spec "$native_real_harness_args_override" "manifest row ${line_no}: invalid native_real_harness_args override"; then
        return 1
      fi
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
    if [[ -n "$example_timeout_override" ]]; then
      EXAMPLE_TO_TIMEOUT_SEC["$example_id"]="$example_timeout_override"
    fi
    if [[ -n "$example_retries_override" ]]; then
      EXAMPLE_TO_RETRIES["$example_id"]="$example_retries_override"
    fi
    if [[ -n "$example_retry_delay_ms_override" ]]; then
      EXAMPLE_TO_RETRY_DELAY_MS["$example_id"]="$example_retry_delay_ms_override"
    fi
    if [[ -n "$max_detected_drop_override" ]]; then
      EXAMPLE_TO_MAX_DETECTED_DROP["$example_id"]="$max_detected_drop_override"
    fi
    if [[ -n "$max_relevant_drop_override" ]]; then
      EXAMPLE_TO_MAX_RELEVANT_DROP["$example_id"]="$max_relevant_drop_override"
    fi
    if [[ -n "$max_coverage_drop_percent_override" ]]; then
      EXAMPLE_TO_MAX_COVERAGE_DROP_PERCENT["$example_id"]="$max_coverage_drop_percent_override"
    fi
    if [[ -n "$max_errors_increase_override" ]]; then
      EXAMPLE_TO_MAX_ERRORS_INCREASE["$example_id"]="$max_errors_increase_override"
    fi
    if [[ -n "$max_detected_drop_percent_override" ]]; then
      EXAMPLE_TO_MAX_DETECTED_DROP_PERCENT["$example_id"]="$max_detected_drop_percent_override"
    fi
    if [[ -n "$max_relevant_drop_percent_override" ]]; then
      EXAMPLE_TO_MAX_RELEVANT_DROP_PERCENT["$example_id"]="$max_relevant_drop_percent_override"
    fi
    if [[ -n "$native_real_harness_override" ]]; then
      EXAMPLE_TO_NATIVE_REAL_HARNESS["$example_id"]="$native_real_harness_override"
    fi
    if [[ -n "$native_mutation_ops_override" ]]; then
      EXAMPLE_TO_NATIVE_MUTATION_OPS["$example_id"]="$native_mutation_ops_override"
    fi
    if [[ -n "$native_real_harness_args_override" ]]; then
      EXAMPLE_TO_NATIVE_REAL_HARNESS_ARGS["$example_id"]="$native_real_harness_args_override"
    fi

    if [[ -n "$max_total_detected_drop_override" ]]; then
      if [[ -z "$manifest_max_total_detected_drop_override" ]]; then
        manifest_max_total_detected_drop_override="$max_total_detected_drop_override"
      elif [[ "$manifest_max_total_detected_drop_override" != "$max_total_detected_drop_override" ]]; then
        echo "Conflicting max_total_detected_drop override in manifest row ${line_no}: ${max_total_detected_drop_override} (already set to ${manifest_max_total_detected_drop_override})" >&2
        return 1
      fi
    fi
    if [[ -n "$max_total_detected_drop_percent_override" ]]; then
      if [[ -z "$manifest_max_total_detected_drop_percent_override" ]]; then
        manifest_max_total_detected_drop_percent_override="$max_total_detected_drop_percent_override"
      elif [[ "$manifest_max_total_detected_drop_percent_override" != "$max_total_detected_drop_percent_override" ]]; then
        echo "Conflicting max_total_detected_drop_percent override in manifest row ${line_no}: ${max_total_detected_drop_percent_override} (already set to ${manifest_max_total_detected_drop_percent_override})" >&2
        return 1
      fi
    fi
    if [[ -n "$max_total_relevant_drop_override" ]]; then
      if [[ -z "$manifest_max_total_relevant_drop_override" ]]; then
        manifest_max_total_relevant_drop_override="$max_total_relevant_drop_override"
      elif [[ "$manifest_max_total_relevant_drop_override" != "$max_total_relevant_drop_override" ]]; then
        echo "Conflicting max_total_relevant_drop override in manifest row ${line_no}: ${max_total_relevant_drop_override} (already set to ${manifest_max_total_relevant_drop_override})" >&2
        return 1
      fi
    fi
    if [[ -n "$max_total_relevant_drop_percent_override" ]]; then
      if [[ -z "$manifest_max_total_relevant_drop_percent_override" ]]; then
        manifest_max_total_relevant_drop_percent_override="$max_total_relevant_drop_percent_override"
      elif [[ "$manifest_max_total_relevant_drop_percent_override" != "$max_total_relevant_drop_percent_override" ]]; then
        echo "Conflicting max_total_relevant_drop_percent override in manifest row ${line_no}: ${max_total_relevant_drop_percent_override} (already set to ${manifest_max_total_relevant_drop_percent_override})" >&2
        return 1
      fi
    fi
    if [[ -n "$max_total_coverage_drop_percent_override" ]]; then
      if [[ -z "$manifest_max_total_coverage_drop_percent_override" ]]; then
        manifest_max_total_coverage_drop_percent_override="$max_total_coverage_drop_percent_override"
      elif [[ "$manifest_max_total_coverage_drop_percent_override" != "$max_total_coverage_drop_percent_override" ]]; then
        echo "Conflicting max_total_coverage_drop_percent override in manifest row ${line_no}: ${max_total_coverage_drop_percent_override} (already set to ${manifest_max_total_coverage_drop_percent_override})" >&2
        return 1
      fi
    fi
    if [[ -n "$max_total_errors_increase_override" ]]; then
      if [[ -z "$manifest_max_total_errors_increase_override" ]]; then
        manifest_max_total_errors_increase_override="$max_total_errors_increase_override"
      elif [[ "$manifest_max_total_errors_increase_override" != "$max_total_errors_increase_override" ]]; then
        echo "Conflicting max_total_errors_increase override in manifest row ${line_no}: ${max_total_errors_increase_override} (already set to ${manifest_max_total_errors_increase_override})" >&2
        return 1
      fi
    fi
  done < "$file"

  if [[ -n "$manifest_max_total_detected_drop_override" ]]; then
    MAX_TOTAL_DETECTED_DROP="$manifest_max_total_detected_drop_override"
  fi
  if [[ -n "$manifest_max_total_detected_drop_percent_override" ]]; then
    MAX_TOTAL_DETECTED_DROP_PERCENT="$manifest_max_total_detected_drop_percent_override"
  fi
  if [[ -n "$manifest_max_total_relevant_drop_override" ]]; then
    MAX_TOTAL_RELEVANT_DROP="$manifest_max_total_relevant_drop_override"
  fi
  if [[ -n "$manifest_max_total_relevant_drop_percent_override" ]]; then
    MAX_TOTAL_RELEVANT_DROP_PERCENT="$manifest_max_total_relevant_drop_percent_override"
  fi
  if [[ -n "$manifest_max_total_coverage_drop_percent_override" ]]; then
    MAX_TOTAL_COVERAGE_DROP_PERCENT="$manifest_max_total_coverage_drop_percent_override"
  fi
  if [[ -n "$manifest_max_total_errors_increase_override" ]]; then
    MAX_TOTAL_ERRORS_INCREASE="$manifest_max_total_errors_increase_override"
  fi

  if [[ ${#AVAILABLE_EXAMPLES[@]} -eq 0 ]]; then
    echo "Example manifest has no usable rows: ${file}" >&2
    return 1
  fi
}


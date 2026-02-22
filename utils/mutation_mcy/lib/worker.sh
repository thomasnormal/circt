#!/usr/bin/env bash
# Shared worker lifecycle helpers for mutation MCY runner.

run_example_worker() {
  local example_id="$1"
  local result_file="$2"
  local design="${EXAMPLE_TO_DESIGN[$example_id]}"
  local top="${EXAMPLE_TO_TOP[$example_id]}"
  local example_generate_count="$GENERATE_COUNT"
  local example_mutations_seed="$MUTATIONS_SEED"
  local example_mutations_modes="$MUTATIONS_MODES"
  local example_mutations_mode_counts="$MUTATIONS_MODE_COUNTS"
  local example_mutations_mode_weights="$MUTATIONS_MODE_WEIGHTS"
  local example_mutations_profiles="$MUTATIONS_PROFILES"
  local example_mutations_cfg="$MUTATIONS_CFG"
  local example_mutations_select="$MUTATIONS_SELECT"
  local example_mutation_limit="$MUTATION_LIMIT"
  local example_timeout_sec="$EXAMPLE_TIMEOUT_SEC"
  local example_retries="$EXAMPLE_RETRIES"
  local example_retry_delay_ms="$EXAMPLE_RETRY_DELAY_MS"
  local design_content_hash=""
  local policy_fingerprint_input=""
  local policy_fingerprint=""
  local example_out_dir=""
  local helper_dir=""
  local fake_test_script=""
  local tests_manifest=""
  local mutations_file=""
  local fake_create_mutated=""
  local run_log=""
  local metrics_file=""
  local retry_result_file="${result_file}.retry"
  local retry_events_file="${result_file}.retry_events"
  local retry_attempts=1
  local retries_used=0
  local retry_reason=""
  local rc=0
  local detected="0"
  local relevant="0"
  local errors="0"
  local coverage="-"
  local coverage_for_gate="0"
  local status="FAIL"
  local gate_failure=""
  local cmd=()
  local max_attempts=1
  local attempt=1
  local retry_sleep_sec=""
  local retry_delay_msg=""
  local native_real_harness_override=""
  local native_real_harness_resolved=""
  local native_real_harness_cmd_script=""
  local native_real_harness_policy_token=""
  local examples_root_policy=""
  local native_real_harness_args_spec="$NATIVE_REAL_HARNESS_ARGS"
  local native_real_harness_args_suffix=""
  local native_mutation_ops_spec="$NATIVE_MUTATION_OPS"
  local native_noop_fallback_marker=""

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
  if [[ -n "${EXAMPLE_TO_TIMEOUT_SEC[$example_id]+x}" ]]; then
    example_timeout_sec="${EXAMPLE_TO_TIMEOUT_SEC[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_RETRIES[$example_id]+x}" ]]; then
    example_retries="${EXAMPLE_TO_RETRIES[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_RETRY_DELAY_MS[$example_id]+x}" ]]; then
    example_retry_delay_ms="${EXAMPLE_TO_RETRY_DELAY_MS[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_NATIVE_REAL_HARNESS[$example_id]+x}" ]]; then
    native_real_harness_override="${EXAMPLE_TO_NATIVE_REAL_HARNESS[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_NATIVE_MUTATION_OPS[$example_id]+x}" ]]; then
    native_mutation_ops_spec="${EXAMPLE_TO_NATIVE_MUTATION_OPS[$example_id]}"
  fi
  if [[ -n "${EXAMPLE_TO_NATIVE_REAL_HARNESS_ARGS[$example_id]+x}" ]]; then
    native_real_harness_args_spec="${EXAMPLE_TO_NATIVE_REAL_HARNESS_ARGS[$example_id]}"
  fi
  max_attempts=$((example_retries + 1))

  if [[ -n "$example_mutations_mode_counts" && -n "$example_mutations_mode_weights" ]]; then
    echo "Resolved mutation mode allocation conflict for ${example_id}: both mode-counts and mode-weights are set." >&2
    return 2
  fi

  if [[ ! -f "$design" ]]; then
    echo "Missing example design for ${example_id}: $design" >&2
    return 2
  fi

  design_content_hash="$(hash_file_sha256 "$design")"
  policy_fingerprint_input="${example_id}"$'\n'"${top}"$'\n'"${design_content_hash}"$'\n'"${example_generate_count}"$'\n'"${example_mutations_seed}"$'\n'"${example_mutations_modes}"$'\n'"${example_mutations_mode_counts}"$'\n'"${example_mutations_mode_weights}"$'\n'"${example_mutations_profiles}"$'\n'"${example_mutations_cfg}"$'\n'"${example_mutations_select}"$'\n'"${example_mutation_limit}"$'\n'"${example_timeout_sec}"$'\n'"${example_retries}"$'\n'"${example_retry_delay_ms}"$'\n'"${SMOKE}"
  if [[ -n "$native_real_harness_override" ]]; then
    native_real_harness_policy_token="$native_real_harness_override"
    examples_root_policy="$(canonicalize_path_for_policy "$EXAMPLES_ROOT")"
    if [[ "$native_real_harness_policy_token" == "$examples_root_policy" ]]; then
      native_real_harness_policy_token="EXAMPLES_ROOT"
    elif [[ "$native_real_harness_policy_token" == "$examples_root_policy/"* ]]; then
      native_real_harness_policy_token="EXAMPLES_ROOT/${native_real_harness_policy_token#"$examples_root_policy/"}"
    fi
    policy_fingerprint_input+=$'\n'"${native_real_harness_policy_token}"
  fi
  if [[ -n "$native_mutation_ops_spec" ]]; then
    policy_fingerprint_input+=$'\n'"${native_mutation_ops_spec}"
  fi
  if [[ -n "$native_real_harness_args_spec" ]]; then
    if ! native_real_harness_args_suffix="$(render_native_real_harness_args_suffix "$native_real_harness_args_spec" "example ${example_id}: invalid native real harness args")"; then
      return 2
    fi
    if [[ -n "$native_real_harness_args_suffix" ]]; then
      policy_fingerprint_input+=$'\n'"${native_real_harness_args_suffix# }"
    fi
  fi
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

  if [[ "$SMOKE" -ne 1 && "$MUTATIONS_BACKEND" == "native" && "$NATIVE_TESTS_MODE" == "real" ]]; then
    if [[ -n "$native_real_harness_override" ]]; then
      if [[ "$native_real_harness_override" == /* ]]; then
        native_real_harness_resolved="$native_real_harness_override"
      else
        native_real_harness_resolved="${EXAMPLES_ROOT}/${native_real_harness_override}"
      fi
      if [[ ! -f "$native_real_harness_resolved" ]]; then
        echo "configured native_real_harness_script is missing or not a file for ${example_id}: ${native_real_harness_resolved}" >&2
        return 2
      fi
      native_real_harness_cmd_script="$(shell_escape_word "$native_real_harness_resolved")"
      printf 'sim_real	bash %s ../mutant.v%s	result.txt	^DETECTED$	^SURVIVED$
' "$native_real_harness_cmd_script" "$native_real_harness_args_suffix" > "$tests_manifest"
    else
      case "$example_id" in
      bitcnt)
        real_test_script="${helper_dir}/real_bitcnt_test.sh"
        cp "$(dirname "$design")/bitcnt_tb.v" "${helper_dir}/bitcnt_tb.v"
        cat > "$real_test_script" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
mutant_path="${1:-../mutant.v}"
helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
iverilog_bin="${IVERILOG:-iverilog}"
vvp_bin="${VVP:-vvp}"
rm -f compile.log sim.out
# bitcnt_tb.v uses identifier `do`, which is accepted in Verilog-2005 but a
# keyword in SystemVerilog; try 2005 first, then 2012 for broader compatibility.
if ! "$iverilog_bin" -g2005 -o sim "${helper_dir}/bitcnt_tb.v" "$mutant_path" > compile.log 2>&1; then
  if ! "$iverilog_bin" -g2012 -o sim "${helper_dir}/bitcnt_tb.v" "$mutant_path" >> compile.log 2>&1; then
    echo DETECTED > result.txt
    exit 0
  fi
fi
if ! "$vvp_bin" -n sim > sim.out 2>&1; then
  echo DETECTED > result.txt
  exit 0
fi
if grep -q 'ERROR' sim.out; then
  echo DETECTED > result.txt
else
  echo SURVIVED > result.txt
fi
EOS
        chmod +x "$real_test_script"
        native_real_harness_cmd_script="$(shell_escape_word "$real_test_script")"
        printf 'sim_real	bash %s ../mutant.v%s	result.txt	^DETECTED$	^SURVIVED$
' "$native_real_harness_cmd_script" "$native_real_harness_args_suffix" > "$tests_manifest"
        ;;
      picorv32_primes)
        real_test_script="${helper_dir}/real_picorv32_primes_test.sh"
        cp "$(dirname "$design")/sim_simple.v" "${helper_dir}/sim_simple.v"
        cp "$(dirname "$design")/sim_simple.hex" "${helper_dir}/sim_simple.hex"
        cp "$design" "${helper_dir}/original_design.v"
        cat > "$real_test_script" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
mutant_path="${1:-../mutant.v}"
helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
iverilog_bin="${IVERILOG:-iverilog}"
vvp_bin="${VVP:-vvp}"

native_tb="${helper_dir}/sim_simple.native.v"
good_hash_file="${helper_dir}/sim_simple.good.md5"

if [[ ! -f "$native_tb" ]]; then
  sed "s#../../sim_simple.hex#${helper_dir}/sim_simple.hex#g" "${helper_dir}/sim_simple.v" > "$native_tb"
fi

if [[ ! -f "$good_hash_file" ]]; then
  if ! "$iverilog_bin" -g2012 -o "${helper_dir}/sim_ref" "$native_tb" "${helper_dir}/original_design.v" > "${helper_dir}/sim_ref.compile.log" 2>&1; then
    echo DETECTED > result.txt
    exit 0
  fi
  if ! (cd "$helper_dir" && "$vvp_bin" -N ./sim_ref +mut=0 > sim_ref.out 2>&1); then
    echo DETECTED > result.txt
    exit 0
  fi
  md5sum "${helper_dir}/sim_ref.out" | awk '{ print $1; }' > "$good_hash_file"
fi

good_md5sum="$(cat "$good_hash_file")"
if ! "$iverilog_bin" -g2012 -o "${helper_dir}/sim_mut" "$native_tb" "$mutant_path" > "${helper_dir}/sim_mut.compile.log" 2>&1; then
  echo DETECTED > result.txt
  exit 0
fi
if ! (cd "$helper_dir" && "$vvp_bin" -N ./sim_mut +mut=1 > sim_mut.out 2>&1); then
  echo DETECTED > result.txt
  exit 0
fi
this_md5sum="$(md5sum "${helper_dir}/sim_mut.out" | awk '{ print $1; }')"
if [[ "$good_md5sum" == "$this_md5sum" ]]; then
  echo SURVIVED > result.txt
else
  echo DETECTED > result.txt
fi
EOS
        chmod +x "$real_test_script"
        native_real_harness_cmd_script="$(shell_escape_word "$real_test_script")"
        printf 'sim_real	bash %s ../mutant.v%s	result.txt	^DETECTED$	^SURVIVED$
' "$native_real_harness_cmd_script" "$native_real_harness_args_suffix" > "$tests_manifest"
        ;;
      *)
        if [[ "$NATIVE_REAL_TESTS_STRICT" -eq 1 ]]; then
          echo "native real tests are required but not configured for ${example_id}" >&2
          return 2
        fi
        echo "warning: native real tests not configured for ${example_id}; falling back to synthetic harness" >&2
        ;;
      esac
    fi
  fi

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
    if [[ "$MUTATIONS_BACKEND" == "yosys" ]]; then
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
    else
      mutations_file="${helper_dir}/native.mutations.txt"
      native_create_mutated="${helper_dir}/native_create_mutated.py"
      native_noop_fallback_marker="${helper_dir}/native_noop_fallback.labels"
      rm -f "$native_noop_fallback_marker"
      if ! generate_native_mutation_plan \
        "$design" "$example_generate_count" "$example_mutations_seed" \
        "$native_mutation_ops_spec" "$mutations_file"; then
        echo "Failed to generate native mutation plan for ${example_id}" >&2
        return 2
      fi
      if [[ ! -f "$NATIVE_CREATE_MUTATED_TEMPLATE" ]]; then
        echo "Missing native mutation template: $NATIVE_CREATE_MUTATED_TEMPLATE" >&2
        return 2
      fi
      cp "$NATIVE_CREATE_MUTATED_TEMPLATE" "$native_create_mutated"
      chmod +x "$native_create_mutated"
      cmd+=(
        --mutations-file "$mutations_file"
        --create-mutated-script "$native_create_mutated"
        --mutant-format v
      )
    fi
  fi

  metrics_file="${example_out_dir}/metrics.tsv"
  run_log="${example_out_dir}/run.log"
  : > "$retry_events_file"
  while true; do
    rm -f "$metrics_file"
    set +e
    if [[ -n "$native_noop_fallback_marker" ]]; then
      export CIRCT_MUT_NATIVE_NOOP_FALLBACK_MARKER="$native_noop_fallback_marker"
    else
      unset CIRCT_MUT_NATIVE_NOOP_FALLBACK_MARKER
    fi
    if [[ "$example_timeout_sec" -gt 0 ]]; then
      "$TIMEOUT_RESOLVED" "$example_timeout_sec" "${cmd[@]}" >"$run_log" 2>&1
      rc=$?
    else
      "${cmd[@]}" >"$run_log" 2>&1
      rc=$?
    fi
    unset CIRCT_MUT_NATIVE_NOOP_FALLBACK_MARKER
    set -e

    if [[ "$rc" -eq 0 ]]; then
      break
    fi
    if [[ "$example_timeout_sec" -gt 0 && "$rc" -eq 124 ]]; then
      echo "Example timeout (${example_id}): exceeded ${example_timeout_sec}s" >&2
      break
    fi
    if [[ "$attempt" -ge "$max_attempts" ]]; then
      break
    fi
    retry_reason="$(classify_retryable_transient_failure_reason "$rc" "$run_log" || true)"
    if [[ -z "$retry_reason" ]]; then
      break
    fi

    if [[ "$example_retry_delay_ms" -gt 0 ]]; then
      printf -v retry_sleep_sec '%d.%03d' "$((example_retry_delay_ms / 1000))" "$((example_retry_delay_ms % 1000))"
      retry_delay_msg=", delay_ms=${example_retry_delay_ms}"
    else
      retry_sleep_sec=""
      retry_delay_msg=""
    fi
    echo "Retrying example (${example_id}): attempt $((attempt + 1))/${max_attempts} after transient launcher failure (rc=${rc}${retry_delay_msg})" >&2
    printf '%s\t%s\t%s\t%s\t%s\n' "$attempt" "$((attempt + 1))" "$rc" "$retry_reason" "${example_retry_delay_ms}" >> "$retry_events_file"
    if [[ -n "$retry_sleep_sec" ]]; then
      sleep "$retry_sleep_sec"
    fi
    attempt=$((attempt + 1))
  done
  retry_attempts="$attempt"
  retries_used=$((retry_attempts - 1))
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

  if [[ "$FAIL_ON_NATIVE_NOOP_FALLBACK" == "1" && -n "$native_noop_fallback_marker" && -s "$native_noop_fallback_marker" ]]; then
    status="FAIL"
    if [[ -n "$gate_failure" ]]; then
      gate_failure+=","
    fi
    gate_failure+="native_noop_fallback"
  fi

  if [[ "$detected" -lt "$MIN_DETECTED" ]]; then
    gate_failure="detected<${MIN_DETECTED}"
  fi
  if [[ "$relevant" -lt "$MIN_RELEVANT" ]]; then
    if [[ -n "$gate_failure" ]]; then
      gate_failure+=","
    fi
    gate_failure+="relevant<${MIN_RELEVANT}"
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
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$status" "$rc" "$detected" "$relevant" "$coverage" "$errors" "$policy_fingerprint" "$coverage_for_gate" "$gate_failure" \
    > "$result_file"
  printf '%s\t%s\n' "$retry_attempts" "$retries_used" > "$retry_result_file"

  return 0
}

remove_running_pid() {
  local pid_to_remove="$1"
  local running_pid=""
  local updated_pids=()
  for running_pid in "${RUNNING_PIDS[@]}"; do
    if [[ "$running_pid" != "$pid_to_remove" ]]; then
      updated_pids+=("$running_pid")
    fi
  done
  RUNNING_PIDS=("${updated_pids[@]}")
}

reap_worker_pid() {
  local finished_pid="$1"
  local wrc=0
  set +e
  wait "$finished_pid"
  wrc=$?
  set -e
  if [[ "$wrc" -ne 0 ]]; then
    echo "Example worker failed (${PID_TO_EXAMPLE[$finished_pid]}) with exit code $wrc" >&2
    worker_failures=1
  fi
  remove_running_pid "$finished_pid"
  unset "PID_TO_EXAMPLE[$finished_pid]"
}

wait_for_any_worker_completion() {
  local finished_pid=""

  while [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
    for finished_pid in "${RUNNING_PIDS[@]}"; do
      if ! kill -0 "$finished_pid" 2>/dev/null; then
        reap_worker_pid "$finished_pid"
        return 0
      fi
    done
    sleep 0.05
  done

  return 0
}

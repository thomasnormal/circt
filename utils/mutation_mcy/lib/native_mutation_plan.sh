#!/usr/bin/env bash
# Shared shell glue for native mutation planning in MCY runner.

generate_native_mutation_plan() {
  local design="$1"
  local generate_count="$2"
  local mutations_seed="$3"
  local native_mutation_ops_spec="$4"
  local output_file="$5"

  if command -v python3 >/dev/null 2>&1 && [[ -f "$NATIVE_MUTATION_PLAN_TOOL" ]]; then
    if python3 "$NATIVE_MUTATION_PLAN_TOOL" \
      --design "$design" \
      --count "$generate_count" \
      --seed "$mutations_seed" \
      --ops-csv "$native_mutation_ops_spec" \
      --out "$output_file"; then
      return 0
    fi
  fi

  local native_ops_all=(
    EQ_TO_NEQ
    NEQ_TO_EQ
    LT_TO_LE
    GT_TO_GE
    LE_TO_LT
    GE_TO_GT
    AND_TO_OR
    OR_TO_AND
    XOR_TO_OR
    UNARY_NOT_DROP
    CONST0_TO_1
    CONST1_TO_0
  )
  local native_ops=("${native_ops_all[@]}")
  if [[ -n "$native_mutation_ops_spec" ]]; then
    local -a native_ops_requested=()
    IFS=',' read -r -a native_ops_requested <<< "$native_mutation_ops_spec"
    native_ops=()
    local native_op_name=""
    for native_op_name in "${native_ops_requested[@]}"; do
      native_op_name="$(trim_whitespace "$native_op_name")"
      if [[ -n "$native_op_name" ]]; then
        native_ops+=("$native_op_name")
      fi
    done
  fi

  : > "$output_file"
  local native_ops_count="${#native_ops[@]}"
  if [[ "$native_ops_count" -eq 0 ]]; then
    return 1
  fi
  local native_seed_offset=$((mutations_seed % native_ops_count))
  local mid=0
  local op_idx=0
  for ((mid=1; mid<=generate_count; ++mid)); do
    op_idx=$(((native_seed_offset + mid - 1) % native_ops_count))
    printf '%d NATIVE_%s\n' "$mid" "${native_ops[$op_idx]}" >> "$output_file"
  done
  return 0
}

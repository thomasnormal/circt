#!/usr/bin/env bash
# Generate mutation list using yosys mutate -list.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: generate_mutations_yosys.sh [options]

Required:
  --design FILE             Input design (.il/.v/.sv)
  --out FILE                Output mutation list file

Optional:
  --top NAME                Top module name (recommended for .v/.sv)
  --count N                 Number of mutations to generate (default: 1000)
  --seed N                  Random seed for mutate (default: 1)
  --yosys PATH              Yosys executable (default: yosys)
  --mode NAME               Mutate mode (repeatable)
                            Concrete: inv,const0,const1,cnot0,cnot1
                            Families: arith,control,balanced,all
  --modes CSV               Comma-separated mutate modes (alternative to repeated --mode)
  --mode-count NAME=COUNT   Explicit mutation count for a mode (repeatable)
  --mode-counts CSV         Comma-separated NAME=COUNT mode allocations
  --profile NAME            Named mutation profile (repeatable)
  --profiles CSV            Comma-separated named mutation profiles
  --cfg KEY=VALUE           Mutate config entry (repeatable, becomes -cfg KEY VALUE)
  --cfgs CSV                Comma-separated KEY=VALUE mutate config entries
  --select EXPR             Additional mutate select expression (repeatable)
  --selects CSV             Comma-separated mutate select expressions
  -h, --help                Show help

Output format:
  Each line in --out is "<id> <mutation-spec>" (MCY-compatible list form).
USAGE
}

DESIGN=""
OUT_FILE=""
TOP=""
COUNT=1000
SEED=1
YOSYS_BIN="yosys"
MODES_CSV=""
MODE_COUNTS_CSV=""
PROFILES_CSV=""
CFGS_CSV=""
SELECTS_CSV=""
declare -a MODE_LIST=()
declare -a MODE_COUNT_LIST=()
declare -a PROFILE_LIST=()
declare -a CFG_LIST=()
declare -a SELECT_LIST=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --design) DESIGN="$2"; shift 2 ;;
    --out) OUT_FILE="$2"; shift 2 ;;
    --top) TOP="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --yosys) YOSYS_BIN="$2"; shift 2 ;;
    --mode) MODE_LIST+=("$2"); shift 2 ;;
    --modes) MODES_CSV="$2"; shift 2 ;;
    --mode-count) MODE_COUNT_LIST+=("$2"); shift 2 ;;
    --mode-counts) MODE_COUNTS_CSV="$2"; shift 2 ;;
    --profile) PROFILE_LIST+=("$2"); shift 2 ;;
    --profiles) PROFILES_CSV="$2"; shift 2 ;;
    --cfg) CFG_LIST+=("$2"); shift 2 ;;
    --cfgs) CFGS_CSV="$2"; shift 2 ;;
    --select) SELECT_LIST+=("$2"); shift 2 ;;
    --selects) SELECTS_CSV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DESIGN" || -z "$OUT_FILE" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi
if [[ ! -f "$DESIGN" ]]; then
  echo "Design file not found: $DESIGN" >&2
  exit 1
fi
if [[ ! "$COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --count value: $COUNT" >&2
  exit 1
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Invalid --seed value: $SEED" >&2
  exit 1
fi
if [[ -n "$MODES_CSV" ]]; then
  IFS=',' read -r -a modes_from_csv <<< "$MODES_CSV"
  for mode in "${modes_from_csv[@]}"; do
    mode="${mode#"${mode%%[![:space:]]*}"}"
    mode="${mode%"${mode##*[![:space:]]}"}"
    [[ -z "$mode" ]] && continue
    MODE_LIST+=("$mode")
  done
fi
if [[ -n "$MODE_COUNTS_CSV" ]]; then
  IFS=',' read -r -a mode_counts_from_csv <<< "$MODE_COUNTS_CSV"
  for mode_count in "${mode_counts_from_csv[@]}"; do
    mode_count="${mode_count#"${mode_count%%[![:space:]]*}"}"
    mode_count="${mode_count%"${mode_count##*[![:space:]]}"}"
    [[ -z "$mode_count" ]] && continue
    MODE_COUNT_LIST+=("$mode_count")
  done
fi
if [[ -n "$PROFILES_CSV" ]]; then
  IFS=',' read -r -a profiles_from_csv <<< "$PROFILES_CSV"
  for profile in "${profiles_from_csv[@]}"; do
    profile="${profile#"${profile%%[![:space:]]*}"}"
    profile="${profile%"${profile##*[![:space:]]}"}"
    [[ -z "$profile" ]] && continue
    PROFILE_LIST+=("$profile")
  done
fi
if [[ -n "$CFGS_CSV" ]]; then
  IFS=',' read -r -a cfgs_from_csv <<< "$CFGS_CSV"
  for cfg in "${cfgs_from_csv[@]}"; do
    cfg="${cfg#"${cfg%%[![:space:]]*}"}"
    cfg="${cfg%"${cfg##*[![:space:]]}"}"
    [[ -z "$cfg" ]] && continue
    CFG_LIST+=("$cfg")
  done
fi
if [[ -n "$SELECTS_CSV" ]]; then
  IFS=',' read -r -a selects_from_csv <<< "$SELECTS_CSV"
  for sel in "${selects_from_csv[@]}"; do
    sel="${sel#"${sel%%[![:space:]]*}"}"
    sel="${sel%"${sel##*[![:space:]]}"}"
    [[ -z "$sel" ]] && continue
    SELECT_LIST+=("$sel")
  done
fi
declare -a PROFILE_MODE_LIST=()
declare -a PROFILE_CFG_LIST=()
declare -a PROFILE_SELECT_LIST=()

mode_family_targets() {
  local mode_name="$1"
  case "$mode_name" in
    arith)
      printf "%s\n" "inv" "const0" "const1"
      ;;
    control)
      printf "%s\n" "cnot0" "cnot1"
      ;;
    balanced|all)
      printf "%s\n" "inv" "const0" "const1" "cnot0" "cnot1"
      ;;
    *)
      printf "%s\n" "$mode_name"
      ;;
  esac
}

append_profile() {
  local profile_name="$1"
  case "$profile_name" in
    arith-depth)
      PROFILE_MODE_LIST+=("arith")
      PROFILE_CFG_LIST+=(
        "weight_pq_w=5"
        "weight_pq_mw=5"
        "weight_pq_b=3"
        "weight_pq_mb=3"
      )
      ;;
    control-depth)
      PROFILE_MODE_LIST+=("control")
      PROFILE_CFG_LIST+=(
        "weight_pq_c=5"
        "weight_pq_mc=5"
        "weight_pq_s=3"
        "weight_pq_ms=3"
      )
      ;;
    balanced-depth)
      PROFILE_MODE_LIST+=("arith" "control")
      PROFILE_CFG_LIST+=(
        "weight_pq_w=4"
        "weight_pq_mw=4"
        "weight_pq_c=4"
        "weight_pq_mc=4"
        "weight_pq_s=2"
        "weight_pq_ms=2"
      )
      ;;
    cover)
      PROFILE_CFG_LIST+=(
        "weight_cover=5"
        "pick_cover_prcnt=80"
      )
      ;;
    none)
      ;;
    *)
      echo "Unknown --profile value: $profile_name (expected arith-depth|control-depth|balanced-depth|cover|none)." >&2
      exit 1
      ;;
  esac
}

for profile in "${PROFILE_LIST[@]}"; do
  append_profile "$profile"
done

declare -a COMBINED_MODE_LIST=()
declare -A MODE_SEEN=()
declare -a FINAL_MODE_LIST=()
COMBINED_MODE_LIST=("${PROFILE_MODE_LIST[@]}" "${MODE_LIST[@]}")
declare -a MODE_COUNT_KEYS=()
declare -A MODE_COUNT_BY_MODE=()
mode_counts_enabled=0
mode_counts_total=0
for mode_count in "${MODE_COUNT_LIST[@]}"; do
  mode_name="${mode_count%%=*}"
  mode_value="${mode_count#*=}"
  if [[ -z "$mode_name" || "$mode_value" == "$mode_count" ]]; then
    echo "Invalid --mode-count entry: $mode_count (expected NAME=COUNT)." >&2
    exit 1
  fi
  mode_name="${mode_name#"${mode_name%%[![:space:]]*}"}"
  mode_name="${mode_name%"${mode_name##*[![:space:]]}"}"
  mode_value="${mode_value#"${mode_value%%[![:space:]]*}"}"
  mode_value="${mode_value%"${mode_value##*[![:space:]]}"}"
  if [[ -z "$mode_name" ]]; then
    echo "Invalid --mode-count entry: $mode_count (empty mode name)." >&2
    exit 1
  fi
  if [[ ! "$mode_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid --mode-count count for $mode_name: $mode_value (expected positive integer)." >&2
    exit 1
  fi
  if [[ -z "${MODE_COUNT_BY_MODE[$mode_name]+x}" ]]; then
    MODE_COUNT_KEYS+=("$mode_name")
  fi
  MODE_COUNT_BY_MODE["$mode_name"]="$mode_value"
  mode_counts_total=$((mode_counts_total + mode_value))
  mode_counts_enabled=1
done
if [[ "$mode_counts_enabled" -eq 1 && "$mode_counts_total" -ne "$COUNT" ]]; then
  echo "Mode-count total ($mode_counts_total) must match --count ($COUNT)." >&2
  exit 1
fi
if [[ "$mode_counts_enabled" -eq 1 ]]; then
  COMBINED_MODE_LIST+=("${MODE_COUNT_KEYS[@]}")
fi
for mode in "${COMBINED_MODE_LIST[@]}"; do
  mode="${mode#"${mode%%[![:space:]]*}"}"
  mode="${mode%"${mode##*[![:space:]]}"}"
  [[ -z "$mode" ]] && continue
  if [[ -n "${MODE_SEEN[$mode]+x}" ]]; then
    continue
  fi
  MODE_SEEN["$mode"]=1
  FINAL_MODE_LIST+=("$mode")
done
if [[ "${#FINAL_MODE_LIST[@]}" -eq 0 ]]; then
  FINAL_MODE_LIST+=("")
fi
MODE_LIST=("${FINAL_MODE_LIST[@]}")

declare -a COMBINED_CFG_LIST=()
declare -A CFG_BY_KEY=()
declare -a CFG_KEY_ORDER=()
COMBINED_CFG_LIST=("${PROFILE_CFG_LIST[@]}" "${CFG_LIST[@]}")
for cfg in "${COMBINED_CFG_LIST[@]}"; do
  key="${cfg%%=*}"
  value="${cfg#*=}"
  if [[ -z "$key" || "$value" == "$cfg" ]]; then
    echo "Invalid --cfg entry: $cfg (expected KEY=VALUE)." >&2
    exit 1
  fi
  if [[ ! "$key" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "Invalid --cfg key: $key (expected [A-Za-z0-9_]+)." >&2
    exit 1
  fi
  if [[ ! "$value" =~ ^-?[0-9]+$ ]]; then
    echo "Invalid --cfg value for $key: $value (expected integer)." >&2
    exit 1
  fi
  if [[ -z "${CFG_BY_KEY[$key]+x}" ]]; then
    CFG_KEY_ORDER+=("$key")
  fi
  CFG_BY_KEY["$key"]="$value"
done
CFG_LIST=()
for key in "${CFG_KEY_ORDER[@]}"; do
  CFG_LIST+=("${key}=${CFG_BY_KEY[$key]}")
done

declare -a COMBINED_SELECT_LIST=()
declare -A SELECT_SEEN=()
declare -a FINAL_SELECT_LIST=()
COMBINED_SELECT_LIST=("${PROFILE_SELECT_LIST[@]}" "${SELECT_LIST[@]}")
for sel in "${COMBINED_SELECT_LIST[@]}"; do
  sel="${sel#"${sel%%[![:space:]]*}"}"
  sel="${sel%"${sel##*[![:space:]]}"}"
  [[ -z "$sel" ]] && continue
  if [[ -n "${SELECT_SEEN[$sel]+x}" ]]; then
    continue
  fi
  SELECT_SEEN["$sel"]=1
  FINAL_SELECT_LIST+=("$sel")
done
SELECT_LIST=("${FINAL_SELECT_LIST[@]}")


mkdir -p "$(dirname "$OUT_FILE")"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

case "$DESIGN" in
  *.il) read_cmd="read_rtlil \"$DESIGN\"" ;;
  *.sv) read_cmd="read_verilog -sv \"$DESIGN\"" ;;
  *.v)  read_cmd="read_verilog \"$DESIGN\"" ;;
  *)
    echo "Unsupported design extension for $DESIGN (expected .il/.v/.sv)." >&2
    exit 1
    ;;
esac

prep_cmd="prep"
if [[ -n "$TOP" ]]; then
  prep_cmd="prep -top $TOP"
fi

mode_count="${#MODE_LIST[@]}"
base_count=0
extra_count=0
if [[ "$mode_counts_enabled" -eq 0 ]]; then
  base_count=$((COUNT / mode_count))
  extra_count=$((COUNT % mode_count))
fi

declare -a MODE_OUT_FILES
declare -a MODE_TARGET_LIST=()
declare -a MODE_TARGET_COUNTS=()

for idx in "${!MODE_LIST[@]}"; do
  mode="${MODE_LIST[$idx]}"
  list_count=0
  if [[ "$mode_counts_enabled" -eq 1 ]]; then
    if [[ -n "${MODE_COUNT_BY_MODE[$mode]+x}" ]]; then
      list_count="${MODE_COUNT_BY_MODE[$mode]}"
    fi
  else
    list_count="$base_count"
    if [[ "$idx" -lt "$extra_count" ]]; then
      list_count=$((list_count + 1))
    fi
  fi
  [[ "$list_count" -le 0 ]] && continue

  mapfile -t family_targets < <(mode_family_targets "$mode")
  family_count="${#family_targets[@]}"
  family_base=$((list_count / family_count))
  family_extra=$((list_count % family_count))
  for family_idx in "${!family_targets[@]}"; do
    family_mode="${family_targets[$family_idx]}"
    family_list_count="$family_base"
    if [[ "$family_idx" -lt "$family_extra" ]]; then
      family_list_count=$((family_list_count + 1))
    fi
    [[ "$family_list_count" -le 0 ]] && continue
    MODE_TARGET_LIST+=("$family_mode")
    MODE_TARGET_COUNTS+=("$family_list_count")
  done
done

for idx in "${!MODE_TARGET_LIST[@]}"; do
  mode="${MODE_TARGET_LIST[$idx]}"
  list_count="${MODE_TARGET_COUNTS[$idx]}"
  script_file="${WORK_DIR}/mutate.${idx}.ys"
  log_file="${WORK_DIR}/mutate.${idx}.log"
  sources_file="${WORK_DIR}/sources.${idx}.txt"
  mode_out_file="${WORK_DIR}/mutations.${idx}.txt"
  MODE_OUT_FILES+=("$mode_out_file")

  mode_arg=""
  if [[ -n "$mode" ]]; then
    mode_arg="-mode $mode"
  fi

  mutate_cmd="mutate -list $list_count -seed $SEED -none"
  for cfg in "${CFG_LIST[@]}"; do
    key="${cfg%%=*}"
    value="${cfg#*=}"
    mutate_cmd+=" -cfg $key $value"
  done
  if [[ -n "$mode_arg" ]]; then
    mutate_cmd+=" $mode_arg"
  fi
  mutate_cmd+=" -o \"$mode_out_file\" -s \"$sources_file\""
  for sel in "${SELECT_LIST[@]}"; do
    mutate_cmd+=" $sel"
  done

  {
    echo "$read_cmd"
    echo "$prep_cmd"
    echo "$mutate_cmd"
  } > "$script_file"

  "$YOSYS_BIN" -ql "$log_file" "$script_file"
  if [[ ! -f "$mode_out_file" ]]; then
    echo "Mutation generation failed: output file missing for mode '${mode:-default}': $mode_out_file" >&2
    exit 1
  fi
done

: > "$OUT_FILE"
declare -A SEEN_SPECS
next_id=1
done_flag=0
for mode_out_file in "${MODE_OUT_FILES[@]}"; do
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line#"${raw_line%%[![:space:]]*}"}"
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    mut_id="${line%%[[:space:]]*}"
    mut_spec="${line#"$mut_id"}"
    mut_spec="${mut_spec#"${mut_spec%%[![:space:]]*}"}"
    [[ -z "$mut_spec" ]] && continue
    if [[ -n "${SEEN_SPECS[$mut_spec]+x}" ]]; then
      continue
    fi
    SEEN_SPECS["$mut_spec"]=1
    printf "%s %s\n" "$next_id" "$mut_spec" >> "$OUT_FILE"
    next_id=$((next_id + 1))
    if [[ "$next_id" -gt "$COUNT" ]]; then
      done_flag=1
      break
    fi
  done < "$mode_out_file"
  if [[ "$done_flag" -eq 1 ]]; then
    break
  fi
done

if [[ ! -s "$OUT_FILE" ]]; then
  echo "Mutation generation failed: generated mutation set is empty: $OUT_FILE" >&2
  exit 1
fi

echo "Generated mutations: $(wc -l < "$OUT_FILE")"
echo "Mutation file: $OUT_FILE"

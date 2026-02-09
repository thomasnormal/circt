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
  --modes CSV               Comma-separated mutate modes (alternative to repeated --mode)
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
CFGS_CSV=""
SELECTS_CSV=""
declare -a MODE_LIST=()
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
if [[ "${#MODE_LIST[@]}" -eq 0 ]]; then
  MODE_LIST+=("")
fi

for cfg in "${CFG_LIST[@]}"; do
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
done

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
base_count=$((COUNT / mode_count))
extra_count=$((COUNT % mode_count))

declare -a MODE_OUT_FILES

for idx in "${!MODE_LIST[@]}"; do
  mode="${MODE_LIST[$idx]}"
  list_count="$base_count"
  if [[ "$idx" -lt "$extra_count" ]]; then
    list_count=$((list_count + 1))
  fi
  [[ "$list_count" -le 0 ]] && continue

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

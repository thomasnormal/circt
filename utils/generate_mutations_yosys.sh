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
  --mode NAME               Optional mutate mode (passed as: -mode NAME)
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
MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --design) DESIGN="$2"; shift 2 ;;
    --out) OUT_FILE="$2"; shift 2 ;;
    --top) TOP="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --yosys) YOSYS_BIN="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
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

mkdir -p "$(dirname "$OUT_FILE")"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

SCRIPT_FILE="${WORK_DIR}/mutate.ys"
LOG_FILE="${WORK_DIR}/mutate.log"
SOURCES_FILE="${WORK_DIR}/sources.txt"

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

mode_arg=""
if [[ -n "$MODE" ]]; then
  mode_arg="-mode $MODE"
fi

{
  echo "$read_cmd"
  echo "$prep_cmd"
  echo "mutate -list $COUNT -seed $SEED -none $mode_arg -o \"$OUT_FILE\" -s \"$SOURCES_FILE\""
} > "$SCRIPT_FILE"

"$YOSYS_BIN" -ql "$LOG_FILE" "$SCRIPT_FILE"

if [[ ! -f "$OUT_FILE" ]]; then
  echo "Mutation generation failed: output file missing: $OUT_FILE" >&2
  exit 1
fi

echo "Generated mutations: $(wc -l < "$OUT_FILE")"
echo "Mutation file: $OUT_FILE"

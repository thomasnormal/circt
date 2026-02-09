#!/usr/bin/env bash
# Create a mutated design from an MCY-style mutation input line using yosys.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: create_mutated_yosys.sh [options]

Create one mutated design from an MCY-compatible input line:
  "<id> <mutation-spec>"

Options:
  -i, --input FILE         Mutation input file (default: input.txt)
  -o, --output FILE        Output design (.il/.v/.sv) (default: mutated.v)
  -d, --design FILE        Original design file (default: ../../database/design.il)
  -s, --script FILE        Temporary yosys script path (default: mutate.ys)
  -c, --ctrl               Emit mutate -ctrl mutsel flow
  -w, --ctrl-width N       mutsel width with --ctrl (default: 8)
      --yosys PATH         Yosys executable (default: $YOSYS or yosys)
  -h, --help               Show help

The yosys log is written to "<script base>.log".
USAGE
}

INPUT_FILE="input.txt"
OUTPUT_FILE="mutated.v"
SCRIPT_FILE="mutate.ys"
DESIGN_FILE="../../database/design.il"
CTRL_WIDTH=8
USE_CTRL=0
YOSYS_BIN="${YOSYS:-yosys}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -i|--input)
      INPUT_FILE="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    -d|--design)
      DESIGN_FILE="$2"
      shift 2
      ;;
    -s|--script)
      SCRIPT_FILE="$2"
      shift 2
      ;;
    -c|--ctrl)
      USE_CTRL=1
      shift
      ;;
    -w|--ctrl-width)
      CTRL_WIDTH="$2"
      shift 2
      ;;
    --yosys)
      YOSYS_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Input mutation file not found: $INPUT_FILE" >&2
  exit 1
fi
if [[ ! -f "$DESIGN_FILE" ]]; then
  echo "Design file not found: $DESIGN_FILE" >&2
  exit 1
fi
if [[ ! "$CTRL_WIDTH" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --ctrl-width value: $CTRL_WIDTH" >&2
  exit 1
fi

READ_CMD=""
case "$DESIGN_FILE" in
  *.il) READ_CMD="read_rtlil $DESIGN_FILE" ;;
  *.sv) READ_CMD="read_verilog -sv $DESIGN_FILE" ;;
  *.v) READ_CMD="read_verilog $DESIGN_FILE" ;;
  *)
    echo "Unsupported design extension in '$DESIGN_FILE' (expected .il/.v/.sv)." >&2
    exit 1
    ;;
esac

WRITE_CMD=""
case "$OUTPUT_FILE" in
  *.v) WRITE_CMD="write_verilog -norename $OUTPUT_FILE" ;;
  *.sv) WRITE_CMD="write_verilog -norename -sv $OUTPUT_FILE" ;;
  *.il) WRITE_CMD="write_rtlil $OUTPUT_FILE" ;;
  *)
    echo "Unsupported output extension in '$OUTPUT_FILE' (expected .il/.v/.sv)." >&2
    exit 1
    ;;
esac

if [[ "$USE_CTRL" -ne 1 && "$CTRL_WIDTH" -ne 8 ]]; then
  echo "Warning: --ctrl-width ignored without --ctrl." >&2
fi

have_mutation=0
have_more_than_one=0
{
  echo "$READ_CMD"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    idx="${line%% *}"
    if [[ "$line" == "$idx" ]]; then
      continue
    fi
    spec="${line#* }"
    if [[ -z "$spec" ]]; then
      continue
    fi
    if [[ "$USE_CTRL" -eq 1 ]]; then
      echo "mutate -ctrl mutsel $CTRL_WIDTH $idx $spec"
    else
      echo "mutate $spec"
      if [[ "$have_mutation" -eq 1 ]]; then
        have_more_than_one=1
      fi
    fi
    have_mutation=1
  done < "$INPUT_FILE"
  echo "$WRITE_CMD"
} > "$SCRIPT_FILE"

if [[ "$have_mutation" -ne 1 ]]; then
  echo "No valid mutation spec found in: $INPUT_FILE" >&2
  exit 1
fi
if [[ "$have_more_than_one" -eq 1 ]]; then
  echo "Warning: multiple mutations in input without --ctrl; later mutations overwrite earlier state." >&2
fi

LOG_FILE="${SCRIPT_FILE%.*}.log"
"$YOSYS_BIN" -ql "$LOG_FILE" "$SCRIPT_FILE"

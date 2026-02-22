#!/usr/bin/env bash
# Unified helper for whole-project refactor continuation workflow.
set -euo pipefail

PLAN_PATH="docs/WHOLE_PROJECT_REFACTOR_PLAN.md"
TODO_PATH="docs/WHOLE_PROJECT_REFACTOR_TODO.md"
MODE="status"

usage() {
  cat <<'USAGE'
usage: utils/refactor_continue.sh [options]

Options:
  --status           Show concise plan/todo continuation status (default)
  --next             Print one recommended next task
  --prompt           Print canonical continuation prompt
  --plan PATH        Override plan file path
  --todo PATH        Override todo file path
  -h, --help         Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --status)
      MODE="status"
      shift
      ;;
    --next)
      MODE="next"
      shift
      ;;
    --prompt)
      MODE="prompt"
      shift
      ;;
    --plan)
      PLAN_PATH="$2"
      shift 2
      ;;
    --todo)
      TODO_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$PLAN_PATH" ]]; then
  echo "plan file not found: $PLAN_PATH" >&2
  exit 1
fi
if [[ ! -f "$TODO_PATH" ]]; then
  echo "todo file not found: $TODO_PATH" >&2
  exit 1
fi

CANONICAL_PROMPT="continue according to ${PLAN_PATH}, keeping track of progress using ${TODO_PATH}"

if [[ "$MODE" == "prompt" ]]; then
  printf '%s\n' "$CANONICAL_PROMPT"
  exit 0
fi

mapfile -t IN_PROGRESS < <(
  awk '
    /^## / { phase=substr($0,4); next }
    /^- \[~\] / {
      item=$0
      sub(/^- \[~\] /, "", item)
      printf "%s\t%s\n", phase, item
    }
  ' "$TODO_PATH"
)

mapfile -t PENDING < <(
  awk '
    /^## / { phase=substr($0,4); next }
    /^- \[ \] / {
      item=$0
      sub(/^- \[ \] /, "", item)
      printf "%s\t%s\n", phase, item
    }
  ' "$TODO_PATH"
)

next_line=""
next_source=""
if [[ ${#IN_PROGRESS[@]} -gt 0 ]]; then
  next_line="${IN_PROGRESS[0]}"
  next_source="in-progress"
elif [[ ${#PENDING[@]} -gt 0 ]]; then
  next_line="${PENDING[0]}"
  next_source="pending"
fi

format_next() {
  local line="$1"
  if [[ -z "$line" ]]; then
    printf 'none\n'
    return
  fi
  local phase="${line%%$'\t'*}"
  local item="${line#*$'\t'}"
  printf '[%s] %s\n' "$phase" "$item"
}

if [[ "$MODE" == "next" ]]; then
  format_next "$next_line"
  exit 0
fi

printf 'Plan: %s\n' "$PLAN_PATH"
printf 'TODO: %s\n' "$TODO_PATH"
printf 'In-progress tasks: %d\n' "${#IN_PROGRESS[@]}"
printf 'Pending tasks: %d\n' "${#PENDING[@]}"
printf 'Recommended next (%s): %s\n' "${next_source:-none}" "$(format_next "$next_line")"
printf 'Canonical prompt: %s\n' "$CANONICAL_PROMPT"

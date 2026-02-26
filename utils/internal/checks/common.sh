#!/usr/bin/env bash
set -euo pipefail

checks_repo_root() {
  local start_dir="$1"
  local root=""

  root="$(git -C "$start_dir" rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -n "$root" ]]; then
    echo "$root"
    return 0
  fi

  # Fallback for when the checks are executed outside a git worktree.
  echo "$(cd "$start_dir/../../.." && pwd)"
}


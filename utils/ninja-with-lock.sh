#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ninja-with-lock.sh [--no-wait] [--timeout SECONDS] [ninja args...]

Run ninja under a per-build-directory lock to avoid concurrent rebuild races in
shared workspaces.

Options:
  --no-wait           Fail immediately if lock is held.
  --timeout SECONDS   Wait up to SECONDS for the lock (default: wait forever).
  -h, --help          Show this help.

Notes:
  - Lock file lives at <build-dir>/.circt-build.lock
  - Build dir is inferred from the last '-C <dir>' in ninja args, else '.'
EOF
}

if ! command -v flock >/dev/null 2>&1; then
  echo "ninja-with-lock.sh requires 'flock' on PATH" >&2
  exit 127
fi

wait_for_lock=1
timeout_secs=""
ninja_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-wait)
      wait_for_lock=0
      shift
      ;;
    --timeout)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --timeout" >&2
        exit 2
      fi
      timeout_secs="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ninja_args+=("$1")
      shift
      ;;
  esac
done

build_dir="."
for ((i = 0; i < ${#ninja_args[@]}; ++i)); do
  if [[ "${ninja_args[i]}" == "-C" && $((i + 1)) -lt ${#ninja_args[@]} ]]; then
    build_dir="${ninja_args[i + 1]}"
  fi
done

mkdir -p "$build_dir"
lock_file="${build_dir%/}/.circt-build.lock"
owner_file="${build_dir%/}/.circt-build.lock.owner"

print_owner() {
  if [[ -f "$owner_file" ]]; then
    echo "owner metadata from $owner_file:" >&2
    sed 's/^/  /' "$owner_file" >&2 || true
  else
    echo "owner metadata unavailable: $owner_file" >&2
  fi
}

write_owner() {
  local tmp_file
  tmp_file="${owner_file}.tmp.$$"
  {
    echo "pid=$$"
    echo "user=$(id -un 2>/dev/null || echo unknown)"
    echo "host=$(hostname 2>/dev/null || echo unknown)"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "cwd=$PWD"
    printf "command=ninja"
    for arg in "${ninja_args[@]}"; do
      printf " %q" "$arg"
    done
    printf "\n"
  } > "$tmp_file"
  mv -f "$tmp_file" "$owner_file"
}

cleanup_owner() {
  rm -f "$owner_file"
}

exec {lock_fd}>"$lock_file"

if ! flock -n "$lock_fd"; then
  echo "build lock is held: $lock_file" >&2
  print_owner
  if [[ $wait_for_lock -eq 0 ]]; then
    exit 73
  fi
  if [[ -n "$timeout_secs" ]]; then
    echo "waiting for lock (timeout=${timeout_secs}s)..." >&2
    flock -w "$timeout_secs" "$lock_fd"
  else
    echo "waiting for lock..." >&2
    flock "$lock_fd"
  fi
fi

write_owner
trap cleanup_owner EXIT INT TERM

ninja "${ninja_args[@]}"

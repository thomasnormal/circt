#!/usr/bin/env bash
# Serialize ninja builds across agents to avoid host-wide OOM storms.
#
# Usage:
#   utils/ninja_with_lock.sh -C build-test circt-sim
#   CIRCT_BUILD_JOBS=4 utils/ninja_with_lock.sh -C build-test circt-verilog
#
# Env:
#   CIRCT_BUILD_LOCKFILE      default: /tmp/circt-build.lock
#   CIRCT_BUILD_JOBS          default: 4 (used when no -j/--jobs is passed)
#   CIRCT_BUILD_LOCK_TIMEOUT  default: 0 (seconds; 0 => wait forever)

set -euo pipefail

LOCKFILE="${CIRCT_BUILD_LOCKFILE:-/tmp/circt-build.lock}"
LOCK_TIMEOUT="${CIRCT_BUILD_LOCK_TIMEOUT:-0}"
DEFAULT_JOBS="${CIRCT_BUILD_JOBS:-4}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 <ninja args...>" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOCKFILE")"

has_jobs_flag=0
for arg in "$@"; do
  case "$arg" in
    -j|--jobs|-j*)
      has_jobs_flag=1
      break
      ;;
  esac
done

ninja_cmd=(ninja)
if [[ "$has_jobs_flag" -eq 0 ]]; then
  ninja_cmd+=(-j "$DEFAULT_JOBS")
fi
ninja_cmd+=("$@")

exec 9>"$LOCKFILE"
if [[ "$LOCK_TIMEOUT" =~ ^[0-9]+$ ]] && [[ "$LOCK_TIMEOUT" -gt 0 ]]; then
  flock -w "$LOCK_TIMEOUT" 9 || {
    echo "[build-lock] timeout waiting for lock: $LOCKFILE" >&2
    exit 73
  }
else
  flock 9
fi

echo "[build-lock] acquired lock=$LOCKFILE pid=$$ jobs=${DEFAULT_JOBS}"
"${ninja_cmd[@]}"

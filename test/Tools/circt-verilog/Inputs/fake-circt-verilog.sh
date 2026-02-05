#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
  if [[ "$arg" == "--timescale=1ns/1ps" ]]; then
    exit 0
  fi
done

echo "missing --timescale=1ns/1ps" >&2
exit 1

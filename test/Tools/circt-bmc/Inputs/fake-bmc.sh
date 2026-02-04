#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
  if [[ "$arg" == "--rising-clocks-only" ]]; then
    echo "BMC_RESULT=UNSAT"
    echo "Bound reached with no violations!"
    exit 0
  fi
done

echo "BMC_RESULT=SAT"
echo "Assertion can be violated!"
exit 0

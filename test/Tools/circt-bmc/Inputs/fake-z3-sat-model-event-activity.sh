#!/usr/bin/env bash
set -euo pipefail

want_model=0
if [[ "${1:-}" == "-model" ]]; then
  want_model=1
  shift
fi

file="${1:-}"
if [[ -z "$file" ]]; then
  echo "missing input file" >&2
  exit 2
fi

echo "sat"
if [[ "$want_model" == "1" ]] && grep -q "(get-model)" "$file"; then
  cat <<'MODEL'
(model
  (define-fun in () (_ BitVec 1) #b0)
  (define-fun in_0 () (_ BitVec 1) #b1)
  (define-fun seq () (_ BitVec 1) #b0)
  (define-fun seq_0 () (_ BitVec 1) #b1)
  (define-fun en () (_ BitVec 1) #b1)
  (define-fun en_0 () (_ BitVec 1) #b1)
)
MODEL
fi

#!/usr/bin/env bash
have_model=0
smt_file=""
for arg in "$@"; do
  if [ "$arg" = "-model" ]; then
    have_model=1
  elif [[ "$arg" != -* ]]; then
    smt_file="$arg"
  fi
done
if [ $have_model -ne 1 ]; then
  echo "missing -model argument" >&2
  exit 1
fi
if [ -z "$smt_file" ] || [ ! -f "$smt_file" ]; then
  echo "missing SMT file" >&2
  exit 1
fi
if ! grep -q "(get-model)" "$smt_file"; then
  echo "missing (get-model) in SMT file" >&2
  exit 1
fi
echo "sat"
cat <<'MODEL'
(model
  (define-fun in () Bool
    true)
  (define-fun in2 () (_ BitVec 8)
    (_ bv5 8))
  (define-fun in3 () (_ BitVec 72)
    (_ bv18446744073709551617 72))
)
model: ok
MODEL

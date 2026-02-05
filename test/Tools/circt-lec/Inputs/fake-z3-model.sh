#!/usr/bin/env bash
have_model=0
for arg in "$@"; do
  if [ "$arg" = "-model" ]; then
    have_model=1
  fi
done
if [ $have_model -ne 1 ]; then
  echo "missing -model argument" >&2
  exit 1
fi
echo "sat"
cat <<'EOF'
(model
  (define-fun in () Bool
    true)
  (define-fun in2 () (_ BitVec 8)
    (_ bv5 8))
  (define-fun in3 () (_ BitVec 72)
    (_ bv18446744073709551617 72))
)
model: ok
EOF

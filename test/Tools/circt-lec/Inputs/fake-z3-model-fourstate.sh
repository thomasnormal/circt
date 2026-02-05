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
  (define-fun fs () (_ BitVec 16)
    #xB380)
)
model: ok
EOF

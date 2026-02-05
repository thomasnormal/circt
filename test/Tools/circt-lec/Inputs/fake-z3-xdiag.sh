#!/usr/bin/env bash
smtfile=""
for arg in "$@"; do
  smtfile="$arg"
done

if [ -z "$smtfile" ]; then
  echo "missing SMT file argument" >&2
  exit 1
fi

if grep -q "circt-lec: assume-known-inputs" "$smtfile" 2>/dev/null; then
  echo "unsat"
  exit 0
fi

echo "sat"
cat <<'EOF'
(model
  (define-fun fs () (_ BitVec 16)
    #x0001)
)
model: ok
EOF

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <sv-tests-dir> <expect-file>" >&2
  exit 2
fi

SV_TESTS_DIR="$1"
EXPECT_FILE="$2"

if [[ ! -d "$SV_TESTS_DIR/tests" ]]; then
  echo "sv-tests directory not found: $SV_TESTS_DIR" >&2
  exit 2
fi

if [[ ! -f "$EXPECT_FILE" ]]; then
  echo "expect file not found: $EXPECT_FILE" >&2
  exit 2
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

all_tests="$tmpdir/all-tests.txt"
expect_names="$tmpdir/expect-names.txt"
missing="$tmpdir/missing.txt"

find "$SV_TESTS_DIR/tests" -name '*.sv' -printf '%f\n' | \
  sed 's/\.sv$//' | sort -u > "$all_tests"

awk -F '\t' '
  {
    name = $1
    gsub(/\r/, "", name)
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
    if (name == "" || name ~ /^#/)
      next
    print name
  }
' "$EXPECT_FILE" | sort -u > "$expect_names"

comm -23 "$expect_names" "$all_tests" > "$missing"

if [[ -s "$missing" ]]; then
  echo "stale sv-tests expect entries in $EXPECT_FILE:" >&2
  sed 's/^/  /' "$missing" >&2
  exit 1
fi

echo "expect lint OK: $EXPECT_FILE"

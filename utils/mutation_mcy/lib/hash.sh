#!/usr/bin/env bash
# Shared hashing helpers for mutation runner scripts.

HASH_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SH="${HASH_LIB_DIR}/../../lib/common.sh"
if [[ -f "$COMMON_SH" ]]; then
  # shellcheck disable=SC1090
  source "$COMMON_SH"
fi

hash_stdin() {
  if declare -F circt_common_hash_stdin >/dev/null 2>&1; then
    if circt_common_hash_stdin; then
      return 0
    fi
  else
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum | awk '{print $1}'
      return 0
    fi
    if command -v shasum >/dev/null 2>&1; then
      shasum -a 256 | awk '{print $1}'
      return 0
    fi
    if command -v openssl >/dev/null 2>&1; then
      openssl dgst -sha256 | awk '{print $NF}'
      return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
      python3 -c 'import hashlib,sys;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'
      return 0
    fi
  fi
  echo "No SHA-256 hashing tool found (sha256sum/shasum/openssl/python3)." >&2
  return 1
}

hash_string() {
  local value="$1"
  printf '%s' "$value" | hash_stdin
}

hash_file() {
  local path="$1"
  if declare -F circt_common_sha256_of >/dev/null 2>&1; then
    local digest
    digest="$(circt_common_sha256_of "$path")"
    case "$digest" in
      "<missing>"|"<unavailable>")
        ;;
      *)
        printf '%s\n' "$digest"
        return 0
        ;;
    esac
  fi
  hash_stdin < "$path"
}

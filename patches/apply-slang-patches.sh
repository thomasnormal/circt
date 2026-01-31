#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

git apply --ignore-whitespace "$script_dir/slang-class-handle-bool-conversion.patch" || true
git apply --ignore-whitespace "$script_dir/slang-trailing-port-comma.patch" || true
git apply --ignore-whitespace "$script_dir/slang-sequence-syntax.patch" || true
git apply --ignore-whitespace "$script_dir/slang-ifdef-expr.patch" || true
# Note: slang-bind-instantiation-def.patch doesn't apply to slang v10
# git apply --ignore-whitespace "$script_dir/slang-bind-instantiation-def.patch" || true
# Bind scope fix: resolve port connection names in bind directive scope per LRM 23.11
git apply --ignore-whitespace "$script_dir/slang-bind-scope.patch" || true
# Bind wildcard fallback: allow implicit .* to resolve in target scope
git apply --ignore-whitespace "$script_dir/slang-bind-wildcard-scope.patch" || true
# AllowVirtualIfaceWithOverride flag for Xcelium compatibility (bind/vif support)
git apply --ignore-whitespace "$script_dir/slang-allow-virtual-iface-override.patch" || true
git apply --ignore-whitespace "$script_dir/slang-relax-string-concat-byte.patch" || true
git apply --ignore-whitespace "$script_dir/slang-allow-class-handle-format.patch" || true
# Allow clocking argument in $past at position 2 (3rd argument) in addition to position 3
git apply --ignore-whitespace "$script_dir/slang-past-clocking-arg.patch" || true

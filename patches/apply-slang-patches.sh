#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

git apply --ignore-whitespace "$script_dir/slang-class-handle-bool-conversion.patch" || true
git apply --ignore-whitespace "$script_dir/slang-trailing-port-comma.patch" || true
git apply --ignore-whitespace "$script_dir/slang-sequence-syntax.patch" || true
git apply --ignore-whitespace "$script_dir/slang-ifdef-expr.patch" || true
# Note: slang-bind-instantiation-def.patch and slang-bind-scope.patch don't apply to slang v10
# git apply --ignore-whitespace "$script_dir/slang-bind-instantiation-def.patch" || true
# git apply --ignore-whitespace "$script_dir/slang-bind-scope.patch" || true
# AllowVirtualIfaceWithOverride flag for Xcelium compatibility (bind/vif support)
git apply --ignore-whitespace "$script_dir/slang-allow-virtual-iface-override.patch" || true

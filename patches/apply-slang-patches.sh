#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

# AllowVirtualIfaceWithOverride flag for Xcelium compatibility (bind/vif support)
# NOTE: Must be applied FIRST -- it's a superset of class-handle-bool-conversion
# and touches the same files as trailing-port-comma and ifdef-expr.
git apply --ignore-whitespace "$script_dir/slang-allow-virtual-iface-override.patch" || true
# slang-class-handle-bool-conversion is now a subset of allow-virtual-iface-override, skip it
# git apply --ignore-whitespace "$script_dir/slang-class-handle-bool-conversion.patch" || true
git apply --ignore-whitespace "$script_dir/slang-trailing-port-comma.patch" || true
git apply --ignore-whitespace "$script_dir/slang-sequence-syntax.patch" || true
git apply --ignore-whitespace "$script_dir/slang-ifdef-expr.patch" || true
# Note: slang-bind-instantiation-def.patch doesn't apply to slang v10
# git apply --ignore-whitespace "$script_dir/slang-bind-instantiation-def.patch" || true
# Bind scope fix: resolve port connection names in bind directive scope per LRM 23.11
git apply --ignore-whitespace "$script_dir/slang-bind-scope.patch" || true
# Bind wildcard fallback: allow implicit .* to resolve in target scope
git apply --ignore-whitespace "$script_dir/slang-bind-wildcard-scope.patch" || true
git apply --ignore-whitespace "$script_dir/slang-relax-string-concat-byte.patch" || true
git apply --ignore-whitespace "$script_dir/slang-allow-class-handle-format.patch" || true
# Allow clocking argument in $past at position 2 (3rd argument) in addition to position 3
git apply --ignore-whitespace "$script_dir/slang-past-clocking-arg.patch" || true
# Trailing comma in system function call args (e.g. $sformatf("fmt",arg,)) -- Xcelium/VCS compat
git apply --ignore-whitespace "$script_dir/slang-trailing-sysarg-comma.patch" || true
# Downgrade nested block comment from error to warning (SPI AVIP compat)
git apply --ignore-whitespace "$script_dir/slang-nested-block-comment.patch" || true
# Downgrade VirtualArgNoParentDefault from error to warning (JTAG AVIP do_compare compat)
git apply --ignore-whitespace "$script_dir/slang-virtual-arg-default.patch" || true
# Allow randomize-with blocks to access caller's class properties (SPI AVIP compat)
git apply --ignore-whitespace "$script_dir/slang-randomize-with-scope.patch" || true

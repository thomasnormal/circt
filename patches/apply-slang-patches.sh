#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

# Patches being upstreamed as PRs to MikePopoloski/slang:
# - slang-trailing-port-comma.patch -> PR #1669
# - slang-ifdef-expr.patch -> PR #1670
# - slang-allow-class-handle-format.patch -> PR #1671
# - slang-bind-scope.patch -> PR #1672

git apply --ignore-whitespace "$script_dir/slang-trailing-port-comma.patch" || true
git apply --ignore-whitespace "$script_dir/slang-sequence-syntax.patch" || true
git apply --ignore-whitespace "$script_dir/slang-ifdef-expr.patch" || true
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
# Downgrade VirtualArgNoParentDefault from error to warning (JTAG AVIP do_compare compat)
git apply --ignore-whitespace "$script_dir/slang-virtual-arg-default.patch" || true
# Allow randomize-with blocks to access caller's class properties (SPI AVIP compat)
git apply --ignore-whitespace "$script_dir/slang-randomize-with-scope.patch" || true
# Allow covergroup iff without parentheses (extension for Xcelium/VCS compat)
git apply --ignore-whitespace "$script_dir/slang-covergroup-iff-noparen.patch" || true
# Allow missing semicolon before endsequence in sequence declarations
git apply --ignore-whitespace "$script_dir/slang-sequence-decl-semicolon.patch" || true
# Skip ifdef/endif inside `define bodies in skipped preprocessor branches (IEEE §22.5.1)
git apply --ignore-whitespace "$script_dir/slang-define-skip-ifdef.patch" || true

#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate baseline manifest JSON contract and emit optional summary."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

FORMAL_LIB = Path(__file__).resolve().parent / "lib"
if str(FORMAL_LIB) not in sys.path:
    sys.path.insert(0, str(FORMAL_LIB))

from baseline_manifest import ManifestValidationError, load_manifest_commands  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate formal baseline manifest schema contract."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    try:
        payload, commands = load_manifest_commands(manifest_path)
    except ManifestValidationError as exc:
        raise SystemExit(str(exc))

    suite_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    max_timeout_secs = 0
    with_cwd = 0
    commands_with_nonzero_expected_rc = 0
    for command in commands:
        suite_counts[command.suite] += 1
        mode_counts[command.mode] += 1
        if command.timeout_secs > max_timeout_secs:
            max_timeout_secs = command.timeout_secs
        if command.cwd:
            with_cwd += 1
        if any(code != 0 for code in command.expected_returncodes):
            commands_with_nonzero_expected_rc += 1

    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": 1,
            "baseline_id": payload.get("baseline_id", ""),
            "generated_at": payload.get("generated_at", ""),
            "commands": len(commands),
            "suite_counts": dict(sorted(suite_counts.items())),
            "mode_counts": dict(sorted(mode_counts.items())),
            "commands_with_cwd": with_cwd,
            "commands_with_nonzero_expected_rc": commands_with_nonzero_expected_rc,
            "max_timeout_secs": max_timeout_secs,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

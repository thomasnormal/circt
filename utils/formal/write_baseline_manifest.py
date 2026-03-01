#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Write a machine-readable formal baseline manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_key_value(raw: str, kind: str) -> tuple[str, str]:
    if "=" not in raw:
        raise SystemExit(f"invalid {kind} entry (expected key=value): {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise SystemExit(f"invalid {kind} entry (empty key): {raw}")
    return key, value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write formal baseline manifest JSON."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output manifest JSON path.",
    )
    parser.add_argument(
        "--baseline-id",
        required=True,
        help="Stable baseline identifier.",
    )
    parser.add_argument(
        "--generated-at",
        default="",
        help="Override generated timestamp (RFC3339 UTC).",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Repeated command entry as key=value (example: suite=sv-tests).",
    )
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        help="Repeated top-level metadata key=value.",
    )
    args = parser.parse_args()

    generated_at = args.generated_at.strip()
    if not generated_at:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    commands: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for token in args.command:
        key, value = parse_key_value(token, "command")
        if key == "begin":
            if current:
                commands.append(current)
                current = {}
            continue
        if key == "end":
            if current:
                commands.append(current)
                current = {}
            continue
        current[key] = value
    if current:
        commands.append(current)

    metadata: dict[str, str] = {}
    for token in args.meta:
        key, value = parse_key_value(token, "meta")
        metadata[key] = value

    payload = {
        "schema_version": 1,
        "baseline_id": args.baseline_id,
        "generated_at": generated_at,
        "metadata": metadata,
        "commands": commands,
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

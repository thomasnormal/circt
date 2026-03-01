#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate formal results JSONL rows against schema contract v1."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import NoReturn

FORMAL_LIB = Path(__file__).resolve().parent / "lib"
if str(FORMAL_LIB) not in sys.path:
    sys.path.insert(0, str(FORMAL_LIB))

from formal_results_schema import (  # noqa: E402
    validate_schema_v1_row,
)


def fail(path: Path, line_no: int, msg: str) -> NoReturn:
    raise SystemExit(f"{path}:{line_no}: {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate formal result schema JSONL rows."
    )
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl).resolve()
    if not jsonl_path.is_file():
        raise SystemExit(f"results file not found: {jsonl_path}")

    status_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    total_rows = 0
    for line_no, line in enumerate(
        jsonl_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except json.JSONDecodeError as exc:
            fail(jsonl_path, line_no, f"invalid JSON: {exc}")
        if not isinstance(payload, dict):
            fail(jsonl_path, line_no, "row must be a JSON object")
        try:
            status, mode, stage = validate_schema_v1_row(payload)
        except ValueError as exc:
            fail(jsonl_path, line_no, str(exc))
        total_rows += 1
        status_counts[status] += 1
        mode_counts[mode] += 1
        stage_counts[stage] += 1

    if total_rows == 0:
        raise SystemExit(f"{jsonl_path}: no rows found")

    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": 1,
            "rows": total_rows,
            "status_counts": dict(sorted(status_counts.items())),
            "mode_counts": dict(sorted(mode_counts.items())),
            "stage_counts": dict(sorted(stage_counts.items())),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

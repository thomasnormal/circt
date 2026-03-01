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
from typing import Any, NoReturn

FORMAL_LIB = Path(__file__).resolve().parent / "lib"
if str(FORMAL_LIB) not in sys.path:
    sys.path.insert(0, str(FORMAL_LIB))

from formal_results_schema import (  # noqa: E402
    ALLOWED_MODES,
    ALLOWED_STAGES,
    ALLOWED_STATUS,
    REQUIRED_FIELDS,
)


def fail(path: Path, line_no: int, msg: str) -> NoReturn:
    raise SystemExit(f"{path}:{line_no}: {msg}")


def expect_string(path: Path, line_no: int, field: str, value: Any) -> str:
    if not isinstance(value, str):
        fail(path, line_no, f"{field} must be a string")
    return value


def expect_nullable_nonnegative_int(
    path: Path, line_no: int, field: str, value: Any
) -> None:
    if value is None:
        return
    if not isinstance(value, int) or value < 0:
        fail(path, line_no, f"{field} must be null or non-negative integer")


def validate_row(path: Path, line_no: int, row: dict[str, Any]) -> tuple[str, str, str]:
    for field in REQUIRED_FIELDS:
        if field not in row:
            fail(path, line_no, f"missing required field: {field}")
    schema_version = row["schema_version"]
    if schema_version != 1:
        fail(path, line_no, "schema_version must be 1")

    suite = expect_string(path, line_no, "suite", row["suite"]).strip()
    mode = expect_string(path, line_no, "mode", row["mode"]).strip().upper()
    case_id = expect_string(path, line_no, "case_id", row["case_id"]).strip()
    case_path = expect_string(path, line_no, "case_path", row["case_path"]).strip()
    status = expect_string(path, line_no, "status", row["status"]).strip().upper()
    reason_code = (
        expect_string(path, line_no, "reason_code", row["reason_code"])
        .strip()
        .upper()
    )
    stage = expect_string(path, line_no, "stage", row["stage"]).strip().lower()
    _solver = expect_string(path, line_no, "solver", row["solver"]).strip()
    _log_path = expect_string(path, line_no, "log_path", row["log_path"]).strip()
    _artifact_dir = (
        expect_string(path, line_no, "artifact_dir", row["artifact_dir"]).strip()
    )

    if not suite:
        fail(path, line_no, "suite must be non-empty")
    if mode not in ALLOWED_MODES:
        fail(path, line_no, f"invalid mode: {mode}")
    if not case_id:
        fail(path, line_no, "case_id must be non-empty")
    if not case_path:
        fail(path, line_no, "case_path must be non-empty")
    if status not in ALLOWED_STATUS:
        fail(path, line_no, f"invalid status: {status}")
    if stage not in ALLOWED_STAGES:
        fail(path, line_no, f"invalid stage: {stage}")
    if not reason_code and status not in {"PASS", "UNKNOWN"}:
        fail(path, line_no, "reason_code must be non-empty for this status")

    expect_nullable_nonnegative_int(
        path, line_no, "solver_time_ms", row["solver_time_ms"]
    )
    expect_nullable_nonnegative_int(
        path, line_no, "frontend_time_ms", row["frontend_time_ms"]
    )
    return status, mode, stage


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
        status, mode, stage = validate_row(jsonl_path, line_no, payload)
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

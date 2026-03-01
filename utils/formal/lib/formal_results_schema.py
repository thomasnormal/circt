#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared formal results schema constants and validators."""

from __future__ import annotations

import re
from typing import Any

REQUIRED_FIELDS = (
    "schema_version",
    "suite",
    "mode",
    "case_id",
    "case_path",
    "status",
    "reason_code",
    "stage",
    "solver",
    "solver_time_ms",
    "frontend_time_ms",
    "log_path",
    "artifact_dir",
)

ALLOWED_MODES = {"BMC", "LEC", "CONNECTIVITY_LEC"}

ALLOWED_STATUS = {
    "PASS",
    "FAIL",
    "ERROR",
    "TIMEOUT",
    "UNKNOWN",
    "SKIP",
    "XFAIL",
    "XPASS",
}

ALLOWED_STAGES = {"frontend", "lowering", "solver", "result", "postprocess"}
REASON_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def expect_string(field: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def expect_nullable_nonnegative_int(field: str, value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be null or non-negative integer")
    return value


def validate_schema_v1_row(row: dict[str, Any]) -> tuple[str, str, str]:
    """Validate one schema v1 row and return normalized status/mode/stage."""

    for field in REQUIRED_FIELDS:
        if field not in row:
            raise ValueError(f"missing required field: {field}")

    schema_version = row["schema_version"]
    if schema_version != 1:
        raise ValueError("schema_version must be 1")

    suite = expect_string("suite", row["suite"]).strip()
    mode = expect_string("mode", row["mode"]).strip().upper()
    case_id = expect_string("case_id", row["case_id"]).strip()
    case_path = expect_string("case_path", row["case_path"]).strip()
    status = expect_string("status", row["status"]).strip().upper()
    reason_code = expect_string("reason_code", row["reason_code"]).strip().upper()
    stage = expect_string("stage", row["stage"]).strip().lower()
    _solver = expect_string("solver", row["solver"]).strip()
    _log_path = expect_string("log_path", row["log_path"]).strip()
    _artifact_dir = expect_string("artifact_dir", row["artifact_dir"]).strip()
    expect_nullable_nonnegative_int("solver_time_ms", row["solver_time_ms"])
    expect_nullable_nonnegative_int("frontend_time_ms", row["frontend_time_ms"])

    if not suite:
        raise ValueError("suite must be non-empty")
    if mode not in ALLOWED_MODES:
        raise ValueError(f"invalid mode: {mode}")
    if not case_id:
        raise ValueError("case_id must be non-empty")
    if not case_path:
        raise ValueError("case_path must be non-empty")
    if status not in ALLOWED_STATUS:
        raise ValueError(f"invalid status: {status}")
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"invalid stage: {stage}")
    if not reason_code and status not in {"PASS", "UNKNOWN"}:
        raise ValueError("reason_code must be non-empty for this status")
    if reason_code and not REASON_CODE_PATTERN.fullmatch(reason_code):
        raise ValueError("reason_code must match [A-Z][A-Z0-9_]* when non-empty")
    return status, mode, stage

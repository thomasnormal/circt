#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared formal result schema helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

FormalCaseRow = tuple[str, str, str, str, str, str]
FormalCaseMetadata = tuple[int | None, int | None, str, str]


def infer_stage(status: str, reason_code: str) -> str:
    status_norm = status.strip().upper()
    reason_norm = reason_code.strip().upper()
    if "SETUP_ERROR" in reason_norm or "NO_FILES" in reason_norm:
        return "frontend"
    if status_norm == "TIMEOUT":
        if "COMPILE_CONTRACT" in reason_norm:
            return "frontend"
        if "FRONTEND" in reason_norm:
            return "frontend"
        return "solver"
    if status_norm in {"ERROR", "FAIL"}:
        if "COMPILE_CONTRACT" in reason_norm:
            return "frontend"
        if "FRONTEND" in reason_norm:
            return "frontend"
        if "SMT" in reason_norm or "Z3" in reason_norm or "LEC" in reason_norm:
            return "solver"
    return "result"


def make_result_row(
    *,
    suite: str,
    mode: str,
    case_id: str,
    case_path: str,
    status: str,
    reason_code: str = "",
    stage: str = "",
    solver: str = "",
    solver_time_ms: int | None = None,
    frontend_time_ms: int | None = None,
    log_path: str = "",
    artifact_dir: str = "",
) -> dict[str, object]:
    stage_value = stage.strip() if stage.strip() else infer_stage(status, reason_code)
    row: dict[str, object] = {
        "schema_version": 1,
        "suite": suite,
        "mode": mode,
        "case_id": case_id,
        "case_path": case_path,
        "status": status.strip().upper(),
        "reason_code": reason_code.strip().upper(),
        "stage": stage_value,
        "solver": solver.strip(),
        "solver_time_ms": solver_time_ms,
        "frontend_time_ms": frontend_time_ms,
        "log_path": log_path,
        "artifact_dir": artifact_dir,
    }
    return row


def write_results_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def sort_case_rows(rows: Iterable[FormalCaseRow]) -> list[FormalCaseRow]:
    return sorted(rows, key=lambda item: (item[1], item[0], item[2]))


def write_results_tsv(path: Path, rows: Iterable[FormalCaseRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sort_case_rows(rows):
            handle.write("\t".join(row) + "\n")


def build_jsonl_rows_from_case_rows(
    rows: Iterable[FormalCaseRow],
    *,
    solver: str = "",
    case_metadata_by_case_id: dict[str, FormalCaseMetadata] | None = None,
) -> list[dict[str, object]]:
    result_rows: list[dict[str, object]] = []
    for status, case_id, case_path, suite, mode, reason_code in sort_case_rows(rows):
        frontend_time_ms: int | None = None
        solver_time_ms: int | None = None
        log_path = ""
        artifact_dir = ""
        if case_metadata_by_case_id is not None:
            (
                frontend_time_ms,
                solver_time_ms,
                log_path,
                artifact_dir,
            ) = case_metadata_by_case_id.get(case_id, (None, None, "", ""))
        result_rows.append(
            make_result_row(
                suite=suite,
                mode=mode,
                case_id=case_id,
                case_path=case_path,
                status=status,
                reason_code=reason_code,
                solver=solver,
                solver_time_ms=solver_time_ms,
                frontend_time_ms=frontend_time_ms,
                log_path=log_path,
                artifact_dir=artifact_dir,
            )
        )
    return result_rows


def write_results_jsonl_from_case_rows(
    path: Path,
    rows: Iterable[FormalCaseRow],
    *,
    solver: str = "",
    case_metadata_by_case_id: dict[str, FormalCaseMetadata] | None = None,
) -> None:
    write_results_jsonl(
        path,
        build_jsonl_rows_from_case_rows(
            rows,
            solver=solver,
            case_metadata_by_case_id=case_metadata_by_case_id,
        ),
    )

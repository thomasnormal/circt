#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Summarize timeout frontier metrics from formal results JSONL rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NoReturn


def fail(path: Path, line_no: int, msg: str) -> NoReturn:
    raise SystemExit(f"{path}:{line_no}: {msg}")


def expect_nonnegative_int_or_none(
    path: Path, line_no: int, field: str, value: Any
) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        fail(path, line_no, f"{field} must be null or non-negative integer")
    return value


def expect_string(path: Path, line_no: int, field: str, value: Any) -> str:
    if not isinstance(value, str):
        fail(path, line_no, f"{field} must be a string")
    return value


def nearest_rank_percentile(values: list[int], percentile: int) -> int | None:
    if not values:
        return None
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)
    sorted_values = sorted(values)
    rank = (percentile * len(sorted_values) + 99) // 100
    index = max(0, min(len(sorted_values) - 1, rank - 1))
    return sorted_values[index]


def write_timeout_reasons_tsv(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["reason_code", "count"])
        for reason_code, count in rows:
            writer.writerow([reason_code, str(count)])


def write_top_solver_cases_tsv(
    path: Path,
    rows: list[tuple[str, str, str, str, int, int, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "case_id",
                "case_path",
                "suite",
                "mode",
                "total_solver_time_ms",
                "rows",
                "timeout_rows",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    str(row[4]),
                    str(row[5]),
                    str(row[6]),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize timeout frontier metrics from formal results JSONL."
    )
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--timeout-reasons-tsv", default="")
    parser.add_argument("--top-solver-cases-tsv", default="")
    parser.add_argument("--top-reasons-limit", type=int, default=10)
    parser.add_argument("--top-cases-limit", type=int, default=10)
    parser.add_argument(
        "--include-nonsolver-timeouts",
        action="store_true",
        help=(
            "Include TIMEOUT rows from non-solver stages in timeout frontier "
            "counts/reason clustering."
        ),
    )
    args = parser.parse_args()

    if args.top_reasons_limit < 1:
        raise SystemExit("--top-reasons-limit must be >= 1")
    if args.top_cases_limit < 1:
        raise SystemExit("--top-cases-limit must be >= 1")

    jsonl_path = Path(args.jsonl).resolve()
    if not jsonl_path.is_file():
        raise SystemExit(f"results file not found: {jsonl_path}")

    rows: list[dict[str, Any]] = []
    solver_times: list[int] = []
    frontend_times: list[int] = []
    timeout_reason_counts: Counter[str] = Counter()
    per_case_solver: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(
        lambda: {"solver_time_ms": 0, "rows": 0, "timeout_rows": 0}
    )

    timeout_rows = 0
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
        rows.append(payload)
        total_rows += 1

        case_id = expect_string(jsonl_path, line_no, "case_id", payload.get("case_id", ""))
        case_path = expect_string(
            jsonl_path, line_no, "case_path", payload.get("case_path", "")
        )
        suite = expect_string(jsonl_path, line_no, "suite", payload.get("suite", ""))
        mode = expect_string(jsonl_path, line_no, "mode", payload.get("mode", ""))
        status = expect_string(
            jsonl_path, line_no, "status", payload.get("status", "")
        ).strip().upper()
        stage = expect_string(
            jsonl_path, line_no, "stage", payload.get("stage", "")
        ).strip().lower()
        reason_code = expect_string(
            jsonl_path, line_no, "reason_code", payload.get("reason_code", "")
        ).strip().upper()
        solver_time_ms = expect_nonnegative_int_or_none(
            jsonl_path, line_no, "solver_time_ms", payload.get("solver_time_ms")
        )
        frontend_time_ms = expect_nonnegative_int_or_none(
            jsonl_path, line_no, "frontend_time_ms", payload.get("frontend_time_ms")
        )

        if solver_time_ms is not None:
            solver_times.append(solver_time_ms)
            case_key = (case_id, case_path, suite, mode)
            per_case_solver[case_key]["solver_time_ms"] += solver_time_ms
            per_case_solver[case_key]["rows"] += 1
        if frontend_time_ms is not None:
            frontend_times.append(frontend_time_ms)

        if status == "TIMEOUT":
            if args.include_nonsolver_timeouts or stage == "solver":
                timeout_rows += 1
                timeout_reason_counts[reason_code or "UNKNOWN_TIMEOUT_REASON"] += 1
                case_key = (case_id, case_path, suite, mode)
                per_case_solver[case_key]["timeout_rows"] += 1

    if total_rows == 0:
        raise SystemExit(f"{jsonl_path}: no rows found")

    top_timeout_reasons = sorted(
        timeout_reason_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[: args.top_reasons_limit]

    top_solver_cases = sorted(
        (
            (
                case_id,
                case_path,
                suite,
                mode,
                stats["solver_time_ms"],
                stats["rows"],
                stats["timeout_rows"],
            )
            for (case_id, case_path, suite, mode), stats in per_case_solver.items()
            if stats["solver_time_ms"] > 0
        ),
        key=lambda item: (-item[4], item[0], item[2], item[3]),
    )[: args.top_cases_limit]

    summary = {
        "schema_version": 1,
        "rows": total_rows,
        "timeout_rows": timeout_rows,
        "timeout_rate": (float(timeout_rows) / float(total_rows)),
        "solver_time_rows": len(solver_times),
        "solver_time_total_ms": sum(solver_times),
        "solver_time_percentiles_ms": {
            "p50": nearest_rank_percentile(solver_times, 50),
            "p90": nearest_rank_percentile(solver_times, 90),
            "p99": nearest_rank_percentile(solver_times, 99),
        },
        "frontend_time_rows": len(frontend_times),
        "frontend_time_total_ms": sum(frontend_times),
        "top_timeout_reasons": [
            {"reason_code": reason_code, "count": count}
            for reason_code, count in top_timeout_reasons
        ],
        "top_solver_cases": [
            {
                "case_id": case_id,
                "case_path": case_path,
                "suite": suite,
                "mode": mode,
                "total_solver_time_ms": total_solver_time_ms,
                "rows": row_count,
                "timeout_rows": timeout_row_count,
            }
            for (
                case_id,
                case_path,
                suite,
                mode,
                total_solver_time_ms,
                row_count,
                timeout_row_count,
            ) in top_solver_cases
        ],
    }
    summary_path = Path(args.summary_json).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.timeout_reasons_tsv:
        write_timeout_reasons_tsv(
            Path(args.timeout_reasons_tsv).resolve(), top_timeout_reasons
        )
    if args.top_solver_cases_tsv:
        write_top_solver_cases_tsv(
            Path(args.top_solver_cases_tsv).resolve(), top_solver_cases
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

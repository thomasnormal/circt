#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build schema-only dashboard inputs from one or more formal JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
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


def write_status_tsv(path: Path, rows: list[tuple[str, str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["suite", "mode", "status", "count"])
        for suite, mode, status, count in rows:
            writer.writerow([suite, mode, status, str(count)])


def write_reason_tsv(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["reason_code", "count"])
        for reason_code, count in rows:
            writer.writerow([reason_code, str(count)])


def write_top_timeout_cases_tsv(
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
        for (
            case_id,
            case_path,
            suite,
            mode,
            total_solver_time_ms,
            row_count,
            timeout_row_count,
        ) in rows:
            writer.writerow(
                [
                    case_id,
                    case_path,
                    suite,
                    mode,
                    str(total_solver_time_ms),
                    str(row_count),
                    str(timeout_row_count),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate formal schema JSONL rows into dashboard summary inputs."
        )
    )
    parser.add_argument(
        "--jsonl",
        action="append",
        required=True,
        help="Input formal-results JSONL file. May be specified multiple times.",
    )
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--status-tsv", default="")
    parser.add_argument("--reason-tsv", default="")
    parser.add_argument("--top-timeout-cases-tsv", default="")
    parser.add_argument("--top-timeout-reasons-tsv", default="")
    parser.add_argument("--top-timeout-cases-limit", type=int, default=20)
    parser.add_argument("--top-timeout-reasons-limit", type=int, default=20)
    parser.add_argument(
        "--include-nonsolver-timeouts",
        action="store_true",
        help="Include non-solver TIMEOUT rows in timeout frontier aggregates.",
    )
    args = parser.parse_args()

    if args.top_timeout_cases_limit < 1:
        raise SystemExit("--top-timeout-cases-limit must be >= 1")
    if args.top_timeout_reasons_limit < 1:
        raise SystemExit("--top-timeout-reasons-limit must be >= 1")

    jsonl_paths: list[Path] = []
    for path_str in args.jsonl:
        path = Path(path_str).resolve()
        if not path.is_file():
            raise SystemExit(f"results file not found: {path}")
        jsonl_paths.append(path)

    total_rows = 0
    status_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    suite_mode_counts: Counter[tuple[str, str]] = Counter()
    suite_mode_status_counts: Counter[tuple[str, str, str]] = Counter()
    reason_counts: Counter[str] = Counter()
    timeout_reason_counts: Counter[str] = Counter()

    solver_times: list[int] = []
    frontend_times: list[int] = []
    timeout_rows = 0

    per_case: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(
        lambda: {"solver_time_ms": 0, "rows": 0, "timeout_rows": 0}
    )

    for path in jsonl_paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            token = line.strip()
            if not token:
                continue
            try:
                payload = json.loads(token)
            except json.JSONDecodeError as exc:
                fail(path, line_no, f"invalid JSON: {exc}")
            if not isinstance(payload, dict):
                fail(path, line_no, "row must be a JSON object")
            try:
                status, mode, stage = validate_schema_v1_row(payload)
            except ValueError as exc:
                fail(path, line_no, str(exc))

            case_id = str(payload["case_id"]).strip()
            case_path = str(payload["case_path"]).strip()
            suite = str(payload["suite"]).strip()
            reason_code = str(payload["reason_code"]).strip().upper()
            solver_time_ms = payload["solver_time_ms"]
            frontend_time_ms = payload["frontend_time_ms"]

            total_rows += 1
            status_counts[status] += 1
            mode_counts[mode] += 1
            stage_counts[stage] += 1
            suite_mode_counts[(suite, mode)] += 1
            suite_mode_status_counts[(suite, mode, status)] += 1
            if reason_code:
                reason_counts[reason_code] += 1

            if solver_time_ms is not None:
                solver_times.append(solver_time_ms)
            if frontend_time_ms is not None:
                frontend_times.append(frontend_time_ms)

            case_key = (case_id, case_path, suite, mode)
            per_case[case_key]["rows"] += 1
            if solver_time_ms is not None:
                per_case[case_key]["solver_time_ms"] += solver_time_ms

            if status == "TIMEOUT":
                if args.include_nonsolver_timeouts or stage == "solver":
                    timeout_rows += 1
                    timeout_reason_counts[reason_code or "UNKNOWN_TIMEOUT_REASON"] += 1
                    per_case[case_key]["timeout_rows"] += 1

    if total_rows == 0:
        raise SystemExit("no rows found across input JSONL files")

    status_rows = sorted(
        (
            suite,
            mode,
            status,
            count,
        )
        for (suite, mode, status), count in suite_mode_status_counts.items()
    )

    reason_rows = sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))

    top_timeout_cases = sorted(
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
            for (case_id, case_path, suite, mode), stats in per_case.items()
            if stats["timeout_rows"] > 0
        ),
        key=lambda item: (-item[6], -item[4], item[0], item[2], item[3]),
    )[: args.top_timeout_cases_limit]

    top_timeout_reasons = sorted(
        timeout_reason_counts.items(), key=lambda item: (-item[1], item[0])
    )[: args.top_timeout_reasons_limit]

    summary = {
        "schema_version": 1,
        "input_files": [str(path) for path in jsonl_paths],
        "rows": total_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "suite_mode_counts": [
            {"suite": suite, "mode": mode, "count": count}
            for (suite, mode), count in sorted(suite_mode_counts.items())
        ],
        "suite_mode_status_counts": [
            {"suite": suite, "mode": mode, "status": status, "count": count}
            for (suite, mode, status), count in sorted(suite_mode_status_counts.items())
        ],
        "reason_code_counts": [
            {"reason_code": reason_code, "count": count}
            for reason_code, count in reason_rows
        ],
        "timeout_rows": timeout_rows,
        "timeout_rate": float(timeout_rows) / float(total_rows),
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
        "top_timeout_cases": [
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
            ) in top_timeout_cases
        ],
    }

    summary_path = Path(args.summary_json).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.status_tsv:
        write_status_tsv(Path(args.status_tsv).resolve(), status_rows)
    if args.reason_tsv:
        write_reason_tsv(Path(args.reason_tsv).resolve(), reason_rows)
    if args.top_timeout_cases_tsv:
        write_top_timeout_cases_tsv(
            Path(args.top_timeout_cases_tsv).resolve(), top_timeout_cases
        )
    if args.top_timeout_reasons_tsv:
        write_reason_tsv(Path(args.top_timeout_reasons_tsv).resolve(), top_timeout_reasons)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

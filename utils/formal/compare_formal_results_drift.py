#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare two formal results JSONL files and classify drift."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, NoReturn


Key = tuple[str, str, str]


def fail(msg: str) -> NoReturn:
    raise SystemExit(msg)


def normalize_enum(raw: Any) -> str:
    return str(raw or "").strip().upper()


def normalize_stage(raw: Any) -> str:
    token = str(raw or "").strip().lower()
    if not token:
        return "result"
    return token


def parse_key(row: dict[str, Any], source: Path, line_no: int) -> Key:
    suite = str(row.get("suite", "")).strip()
    mode = str(row.get("mode", "")).strip()
    case_id = str(row.get("case_id", "")).strip()
    if not suite or not mode or not case_id:
        fail(
            f"{source}:{line_no}: missing required key fields "
            f"(suite/mode/case_id)"
        )
    return (suite, mode, case_id)


def load_rows(path: Path) -> dict[Key, dict[str, Any]]:
    if not path.is_file():
        fail(f"results file not found: {path}")
    rows: dict[Key, dict[str, Any]] = {}
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        token = line.strip()
        if not token:
            continue
        try:
            raw = json.loads(token)
        except json.JSONDecodeError as exc:
            fail(f"{path}:{line_no}: invalid JSON: {exc}")
        if not isinstance(raw, dict):
            fail(f"{path}:{line_no}: expected JSON object row")
        key = parse_key(raw, path, line_no)
        # Keep last row on duplicate key to make drift comparison deterministic.
        rows[key] = {
            "status": normalize_enum(raw.get("status")),
            "reason_code": normalize_enum(raw.get("reason_code")),
            "stage": normalize_stage(raw.get("stage")),
            "case_path": str(raw.get("case_path", "")).strip(),
        }
    return rows


def classify_drift(
    baseline: dict[str, Any] | None, candidate: dict[str, Any] | None
) -> str:
    if baseline is None:
        return "NEW_CASE"
    if candidate is None:
        return "MISSING_CASE"
    if baseline["status"] != candidate["status"]:
        return "STATUS_DRIFT"
    if baseline["reason_code"] != candidate["reason_code"]:
        return "REASON_DRIFT"
    if baseline["stage"] != candidate["stage"]:
        return "STAGE_DRIFT"
    return "NO_DRIFT"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline/candidate formal JSONL outputs."
    )
    parser.add_argument("--reference-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--fail-on-status-drift", action="store_true")
    parser.add_argument("--fail-on-reason-drift", action="store_true")
    parser.add_argument("--fail-on-stage-drift", action="store_true")
    parser.add_argument("--fail-on-missing-case", action="store_true")
    parser.add_argument("--fail-on-new-case", action="store_true")
    args = parser.parse_args()

    reference_path = Path(args.reference_jsonl).resolve()
    candidate_path = Path(args.candidate_jsonl).resolve()
    out_tsv_path = Path(args.out_tsv).resolve()
    summary_json_path = (
        Path(args.summary_json).resolve() if args.summary_json else None
    )

    reference_rows = load_rows(reference_path)
    candidate_rows = load_rows(candidate_path)
    all_keys = sorted(set(reference_rows) | set(candidate_rows))

    counts: Counter[str] = Counter()
    out_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "classification\tsuite\tmode\tcase_id\t"
            "reference_status\tcandidate_status\t"
            "reference_reason_code\tcandidate_reason_code\t"
            "reference_stage\tcandidate_stage\t"
            "reference_case_path\tcandidate_case_path\n"
        )
        for key in all_keys:
            baseline = reference_rows.get(key)
            candidate = candidate_rows.get(key)
            classification = classify_drift(baseline, candidate)
            counts[classification] += 1
            suite, mode, case_id = key
            handle.write(
                "\t".join(
                    (
                        classification,
                        suite,
                        mode,
                        case_id,
                        (baseline or {}).get("status", ""),
                        (candidate or {}).get("status", ""),
                        (baseline or {}).get("reason_code", ""),
                        (candidate or {}).get("reason_code", ""),
                        (baseline or {}).get("stage", ""),
                        (candidate or {}).get("stage", ""),
                        (baseline or {}).get("case_path", ""),
                        (candidate or {}).get("case_path", ""),
                    )
                )
            )
            handle.write("\n")

    summary = {
        "schema_version": 1,
        "reference_jsonl": str(reference_path),
        "candidate_jsonl": str(candidate_path),
        "total_cases": len(all_keys),
        "counts": {
            "NO_DRIFT": counts["NO_DRIFT"],
            "STATUS_DRIFT": counts["STATUS_DRIFT"],
            "REASON_DRIFT": counts["REASON_DRIFT"],
            "STAGE_DRIFT": counts["STAGE_DRIFT"],
            "MISSING_CASE": counts["MISSING_CASE"],
            "NEW_CASE": counts["NEW_CASE"],
        },
    }
    if summary_json_path is not None:
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        summary_json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    should_fail = False
    if args.fail_on_status_drift and counts["STATUS_DRIFT"] > 0:
        should_fail = True
    if args.fail_on_reason_drift and counts["REASON_DRIFT"] > 0:
        should_fail = True
    if args.fail_on_stage_drift and counts["STAGE_DRIFT"] > 0:
        should_fail = True
    if args.fail_on_missing_case and counts["MISSING_CASE"] > 0:
        should_fail = True
    if args.fail_on_new_case and counts["NEW_CASE"] > 0:
        should_fail = True

    if should_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Project formal result schema JSONL rows into TSV output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_FIELDS = (
    "status",
    "case_id",
    "case_path",
    "suite",
    "mode",
    "reason_code",
    "stage",
    "solver",
)


def parse_fields(raw: str) -> list[str]:
    if not raw.strip():
        return list(DEFAULT_FIELDS)
    fields: list[str] = []
    for token in raw.split(","):
        field = token.strip()
        if not field:
            continue
        fields.append(field)
    if not fields:
        return list(DEFAULT_FIELDS)
    return fields


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"{path}:{line_no}: expected JSON object row")
        rows.append(payload)
    return rows


def stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project formal result schema JSONL rows into TSV."
    )
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument(
        "--fields",
        default=",".join(DEFAULT_FIELDS),
        help="Comma-separated projection field list.",
    )
    parser.add_argument(
        "--sort-key",
        default="case_id,status,case_path",
        help="Comma-separated field list used for row sort order.",
    )
    parser.add_argument(
        "--include-header",
        action="store_true",
        help="Emit a header row with selected field names.",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl).resolve()
    out_tsv_path = Path(args.out_tsv).resolve()
    if not jsonl_path.is_file():
        raise SystemExit(f"results file not found: {jsonl_path}")

    projection_fields = parse_fields(args.fields)
    sort_fields = parse_fields(args.sort_key)
    rows = load_jsonl(jsonl_path)
    if not rows:
        raise SystemExit(f"{jsonl_path}: no rows found")

    def sort_tuple(row: dict[str, object]) -> tuple[str, ...]:
        return tuple(stringify(row.get(field, "")) for field in sort_fields)

    rows.sort(key=sort_tuple)

    out_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        if args.include_header:
            writer.writerow(projection_fields)
        for row in rows:
            writer.writerow([stringify(row.get(field, "")) for field in projection_fields])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate declarative formal-suite manifest TSV files."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import Iterable

EXPECTED_HEADER = [
    "suite_id",
    "default_root",
    "bmc_runner",
    "bmc_args",
    "lec_runner",
    "lec_args",
    "profiles",
    "notes",
]

PROFILE_SET = {"smoke", "nightly", "full"}


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def iter_rows(path: pathlib.Path) -> Iterable[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0].strip().startswith("#"):
                continue
            yield row


def validate_manifest(path: pathlib.Path, repo_root: pathlib.Path) -> int:
    rows = list(iter_rows(path))
    if not rows:
        die(f"manifest is empty: {path}")

    header = rows[0]
    if header != EXPECTED_HEADER:
        die(
            f"invalid header in {path}; expected {EXPECTED_HEADER}, got {header}"
        )

    seen_ids: set[str] = set()
    data_rows = rows[1:]
    for idx, row in enumerate(data_rows, start=2):
        if len(row) != len(EXPECTED_HEADER):
            die(
                f"invalid column count in {path}:{idx}; "
                f"expected {len(EXPECTED_HEADER)}, got {len(row)}"
            )

        suite_id = row[0].strip()
        default_root = row[1].strip()
        bmc_runner = row[2].strip()
        lec_runner = row[4].strip()
        profiles = [p.strip() for p in row[6].split(",") if p.strip()]

        if not suite_id:
            die(f"empty suite_id in {path}:{idx}")
        if suite_id in seen_ids:
            die(f"duplicate suite_id '{suite_id}' in {path}:{idx}")
        seen_ids.add(suite_id)

        if not default_root or default_root == "-":
            die(f"invalid default_root for suite '{suite_id}' in {path}:{idx}")

        for runner_field, runner_value in (("bmc_runner", bmc_runner), ("lec_runner", lec_runner)):
            if runner_value == "-":
                continue
            runner_path = repo_root / runner_value
            if not runner_path.exists():
                die(
                    f"runner path not found for suite '{suite_id}' "
                    f"field '{runner_field}' in {path}:{idx}: {runner_value}"
                )

        if not profiles:
            die(f"missing profiles for suite '{suite_id}' in {path}:{idx}")
        for p in profiles:
            if p not in PROFILE_SET:
                die(
                    f"invalid profile '{p}' for suite '{suite_id}' "
                    f"in {path}:{idx}; expected subset of {sorted(PROFILE_SET)}"
                )

    print(f"ok: {path} rows={len(data_rows)}")
    return len(data_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests",
        nargs="+",
        help="Manifest TSV path(s) to validate",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = pathlib.Path(__file__).resolve().parent.parent

    total_rows = 0
    for manifest in args.manifests:
        path = pathlib.Path(manifest)
        if not path.is_absolute():
            path = (repo_root / manifest).resolve()
        if not path.exists():
            die(f"manifest not found: {manifest}")
        total_rows += validate_manifest(path, repo_root)

    print(f"validated_manifests={len(args.manifests)} total_rows={total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

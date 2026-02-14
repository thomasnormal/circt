#!/usr/bin/env python3
"""Check drift for OpenTitan FPV objective parity artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PAYLOAD_FIELDS = (
    "objective_class",
    "target_name",
    "case_id",
    "objective_key",
    "bmc",
    "lec",
    "bmc_evidence",
    "lec_evidence",
    "bmc_reason",
    "lec_reason",
)


@dataclass(frozen=True)
class ParityRow:
    objective_id: str
    kind: str
    target_name: str
    payload: dict[str, str]


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-parity-tsv", required=True)
    parser.add_argument("--current-parity-tsv", required=True)
    parser.add_argument("--out-drift-tsv", default="")
    parser.add_argument("--allowlist-file", default="")
    parser.add_argument("--row-allowlist-file", default="")
    parser.add_argument("--fail-on-drift", action="store_true")
    return parser.parse_args()


def load_allowlist(path: Path) -> tuple[set[str], list[str], list[re.Pattern[str]]]:
    exact: set[str] = set()
    prefixes: list[str] = []
    regex_rules: list[re.Pattern[str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            mode = "exact"
            payload = line
            if ":" in line:
                mode, payload = line.split(":", 1)
                mode = mode.strip()
                payload = payload.strip()
            if not payload:
                fail(f"invalid allowlist row {line_no}: empty pattern")
            if mode == "exact":
                exact.add(payload)
            elif mode == "prefix":
                prefixes.append(payload)
            elif mode == "regex":
                try:
                    regex_rules.append(re.compile(payload))
                except re.error as exc:
                    fail(
                        f"invalid allowlist row {line_no}: bad regex '{payload}': {exc}"
                    )
            else:
                fail(
                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
                    "(expected exact|prefix|regex)"
                )
    return exact, prefixes, regex_rules


def is_allowlisted(
    token: str, exact: set[str], prefixes: list[str], regex_rules: list[re.Pattern[str]]
) -> bool:
    if token in exact:
        return True
    for prefix in prefixes:
        if token.startswith(prefix):
            return True
    for pattern in regex_rules:
        if pattern.search(token):
            return True
    return False


def read_parity_rows(path: Path) -> dict[tuple[str, str], ParityRow]:
    if not path.is_file():
        fail(f"parity TSV not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"parity TSV missing header row: {path}")
        required = {"objective_id", "kind"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            fail(
                f"parity TSV missing required columns {missing}: {path} "
                f"(found: {reader.fieldnames})"
            )
        out: dict[tuple[str, str], ParityRow] = {}
        for idx, row in enumerate(reader, start=2):
            objective_id = (row.get("objective_id") or "").strip()
            kind = (row.get("kind") or "").strip()
            if not objective_id or not kind:
                continue
            key = (objective_id, kind)
            if key in out:
                fail(
                    f"duplicate objective parity key '{objective_id}::{kind}' in {path} "
                    f"row {idx}"
                )
            payload: dict[str, str] = {}
            for field in PAYLOAD_FIELDS:
                payload[field] = (row.get(field) or "").strip()
            target_name = payload.get("target_name", "")
            out[key] = ParityRow(
                objective_id=objective_id,
                kind=kind,
                target_name=target_name,
                payload=payload,
            )
    return out


def serialize_payload(payload: dict[str, str]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def write_drift_rows(
    path: Path, rows: list[tuple[str, str, str, str, str, str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "objective_id",
                "kind",
                "target_name",
                "drift_kind",
                "baseline",
                "current",
                "allowlisted",
            ]
        )
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_parity_tsv).resolve()
    current_path = Path(args.current_parity_tsv).resolve()
    out_path = Path(args.out_drift_tsv).resolve() if args.out_drift_tsv else None

    allow_exact: set[str] = set()
    allow_prefix: list[str] = []
    allow_regex: list[re.Pattern[str]] = []
    if args.allowlist_file:
        allow_exact, allow_prefix, allow_regex = load_allowlist(
            Path(args.allowlist_file).resolve()
        )

    row_allow_exact: set[str] = set()
    row_allow_prefix: list[str] = []
    row_allow_regex: list[re.Pattern[str]] = []
    if args.row_allowlist_file:
        row_allow_exact, row_allow_prefix, row_allow_regex = load_allowlist(
            Path(args.row_allowlist_file).resolve()
        )

    baseline_rows = read_parity_rows(baseline_path)
    current_rows = read_parity_rows(current_path)
    all_keys = sorted(set(baseline_rows.keys()) | set(current_rows.keys()))

    drift_rows: list[tuple[str, str, str, str, str, str, str]] = []
    non_allowlisted_rows: list[tuple[str, str, str, str, str, str, str]] = []

    for key in all_keys:
        baseline = baseline_rows.get(key)
        current = current_rows.get(key)
        objective_id, kind = key
        target_name = ""
        drift_kind = ""
        baseline_value = "<absent>"
        current_value = "<absent>"
        if baseline is None and current is not None:
            drift_kind = "new_row"
            target_name = current.target_name
            current_value = serialize_payload(current.payload)
        elif baseline is not None and current is None:
            drift_kind = "missing_row"
            target_name = baseline.target_name
            baseline_value = serialize_payload(baseline.payload)
        elif baseline is not None and current is not None:
            if baseline.payload == current.payload:
                continue
            drift_kind = "payload_changed"
            target_name = current.target_name or baseline.target_name
            baseline_value = serialize_payload(baseline.payload)
            current_value = serialize_payload(current.payload)
        else:
            continue

        row_key = f"{objective_id}::{kind}::{drift_kind}"
        allowlisted = is_allowlisted(
            row_key, row_allow_exact, row_allow_prefix, row_allow_regex
        ) or any(
            is_allowlisted(token, allow_exact, allow_prefix, allow_regex)
            for token in (row_key, f"{objective_id}::{kind}", objective_id, target_name)
        )
        row = (
            objective_id,
            kind,
            target_name,
            drift_kind,
            baseline_value,
            current_value,
            "1" if allowlisted else "0",
        )
        drift_rows.append(row)
        if not allowlisted:
            non_allowlisted_rows.append(row)

    if out_path is not None:
        write_drift_rows(out_path, drift_rows)

    if non_allowlisted_rows:
        print("opentitan fpv objective parity drift detected", file=sys.stderr)
        sample = ", ".join(
            f"{row[0]}::{row[1]}::{row[3]}" for row in non_allowlisted_rows[:5]
        )
        if sample:
            print(sample, file=sys.stderr)
        if args.fail_on_drift:
            raise SystemExit(1)
    else:
        print("opentitan fpv objective parity drift check passed", file=sys.stderr)


if __name__ == "__main__":
    main()

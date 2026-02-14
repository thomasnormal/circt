#!/usr/bin/env python3
"""Check rule-level OpenTitan connectivity status parity between BMC and LEC.

This compares per-rule status summary TSV artifacts emitted by:
- utils/run_opentitan_connectivity_circt_bmc.py
- utils/run_opentitan_connectivity_circt_lec.py

The checker reports deterministic rule-level parity drift rows and supports
allowlist filtering by `rule_id` or `rule_id::kind` tokens.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


CASE_FIELDS = (
    "case_total",
    "case_pass",
    "case_fail",
    "case_xfail",
    "case_xpass",
    "case_error",
    "case_skip",
)


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bmc-status-summary",
        required=True,
        help="Per-rule connectivity BMC status summary TSV.",
    )
    parser.add_argument(
        "--lec-status-summary",
        required=True,
        help="Per-rule connectivity LEC status summary TSV.",
    )
    parser.add_argument(
        "--allowlist-file",
        default="",
        help=(
            "Optional allowlist file. Each non-comment line is exact:<token>, "
            "prefix:<prefix>, regex:<pattern>, or bare exact. Supported tokens are "
            "<rule_id> and <rule_id>::<kind>."
        ),
    )
    parser.add_argument(
        "--out-parity-tsv",
        default="",
        help="Optional output TSV path for parity drift rows.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero when non-allowlisted parity mismatches are detected.",
    )
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


def read_status_summary(path: Path, lane_name: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        fail(f"{lane_name} status summary file not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"{lane_name} status summary missing header row: {path}")
        required = {"rule_id", *CASE_FIELDS}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            fail(
                f"{lane_name} status summary missing required columns {missing}: {path} "
                f"(found: {reader.fieldnames})"
            )
        out: dict[str, dict[str, str]] = {}
        for idx, row in enumerate(reader, start=2):
            rule_id = (row.get("rule_id") or "").strip()
            if not rule_id:
                continue
            if rule_id in out:
                fail(f"duplicate rule_id '{rule_id}' in {path} row {idx}")
            out[rule_id] = {field: (row.get(field) or "").strip() for field in CASE_FIELDS}
    return out


def emit_parity_tsv(path: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["rule_id", "kind", "bmc", "lec", "allowlisted"])
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    bmc_path = Path(args.bmc_status_summary).resolve()
    lec_path = Path(args.lec_status_summary).resolve()

    allow_exact: set[str] = set()
    allow_prefix: list[str] = []
    allow_regex: list[re.Pattern[str]] = []
    if args.allowlist_file:
        allow_exact, allow_prefix, allow_regex = load_allowlist(
            Path(args.allowlist_file).resolve()
        )

    bmc = read_status_summary(bmc_path, "connectivity BMC")
    lec = read_status_summary(lec_path, "connectivity LEC")

    parity_rows: list[tuple[str, str, str, str, str]] = []
    non_allowlisted_rows: list[tuple[str, str, str, str, str]] = []

    def add_row(rule_id: str, kind: str, bmc_value: str, lec_value: str) -> None:
        tokens = (f"{rule_id}::{kind}", rule_id)
        allowlisted = any(
            is_allowlisted(token, allow_exact, allow_prefix, allow_regex)
            for token in tokens
        )
        row = (rule_id, kind, bmc_value, lec_value, "1" if allowlisted else "0")
        parity_rows.append(row)
        if not allowlisted:
            non_allowlisted_rows.append(row)

    bmc_rules = set(bmc.keys())
    lec_rules = set(lec.keys())

    for rule_id in sorted(bmc_rules - lec_rules):
        add_row(rule_id, "missing_in_lec", "present", "absent")
    for rule_id in sorted(lec_rules - bmc_rules):
        add_row(rule_id, "missing_in_bmc", "absent", "present")

    for rule_id in sorted(bmc_rules.intersection(lec_rules)):
        bmc_row = bmc[rule_id]
        lec_row = lec[rule_id]
        for kind in CASE_FIELDS:
            bmc_value = bmc_row.get(kind, "")
            lec_value = lec_row.get(kind, "")
            if bmc_value != lec_value:
                add_row(rule_id, kind, bmc_value, lec_value)

    if args.out_parity_tsv:
        emit_parity_tsv(Path(args.out_parity_tsv).resolve(), parity_rows)

    if non_allowlisted_rows:
        sample = ", ".join(
            f"{rule_id}:{kind}" for rule_id, kind, _, _, _ in non_allowlisted_rows[:6]
        )
        if len(non_allowlisted_rows) > 6:
            sample += ", ..."
        message = (
            "opentitan connectivity status parity mismatches detected: "
            f"rows={len(non_allowlisted_rows)} sample=[{sample}] "
            f"bmc={bmc_path} lec={lec_path}"
        )
        if args.fail_on_mismatch:
            print(message, file=sys.stderr)
            raise SystemExit(1)
        print(f"warning: {message}", file=sys.stderr)
        return

    print(
        "opentitan connectivity status parity check passed: "
        f"rules_bmc={len(bmc)} rules_lec={len(lec)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

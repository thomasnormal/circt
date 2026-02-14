#!/usr/bin/env python3
"""Check OpenTitan connectivity contract-fingerprint parity between BMC and LEC.

This compares resolved-contract artifacts emitted by:
- utils/run_opentitan_connectivity_circt_bmc.py
- utils/run_opentitan_connectivity_circt_lec.py

Each resolved-contract row contributes a parity token:
  <case_id>::<contract_fingerprint>
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bmc-resolved-contracts",
        required=True,
        help="Resolved-contract artifact from OpenTitan connectivity BMC lane.",
    )
    parser.add_argument(
        "--lec-resolved-contracts",
        required=True,
        help="Resolved-contract artifact from OpenTitan connectivity LEC lane.",
    )
    parser.add_argument(
        "--allowlist-file",
        default="",
        help=(
            "Optional allowlist file. Each non-comment line is exact:<token>, "
            "prefix:<prefix>, regex:<pattern>, or bare exact. Supported tokens are "
            "<case_id>::<contract_fingerprint>, <case_id>, and <contract_fingerprint>."
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


def read_resolved_contracts(path: Path, lane_name: str) -> dict[str, tuple[str, str]]:
    if not path.is_file():
        fail(f"{lane_name} resolved-contract artifact not found: {path}")
    out: dict[str, tuple[str, str]] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            fail(
                f"{lane_name} resolved-contract artifact malformed row {line_no}: "
                f"expected >=3 columns"
            )
        case_id = parts[0].strip()
        fingerprint = parts[-1].strip()
        if not case_id or not fingerprint or fingerprint == "-":
            continue
        token = f"{case_id}::{fingerprint}"
        if token in out:
            fail(f"duplicate resolved-contract token '{token}' in {path} row {line_no}")
        out[token] = (case_id, fingerprint)
    return out


def emit_parity_tsv(path: Path, rows: list[tuple[str, str, str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["case_id", "contract_fingerprint", "kind", "bmc", "lec", "allowlisted"]
        )
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    bmc_path = Path(args.bmc_resolved_contracts).resolve()
    lec_path = Path(args.lec_resolved_contracts).resolve()

    allow_exact: set[str] = set()
    allow_prefix: list[str] = []
    allow_regex: list[re.Pattern[str]] = []
    if args.allowlist_file:
        allow_exact, allow_prefix, allow_regex = load_allowlist(
            Path(args.allowlist_file).resolve()
        )

    bmc = read_resolved_contracts(bmc_path, "connectivity BMC")
    lec = read_resolved_contracts(lec_path, "connectivity LEC")

    parity_rows: list[tuple[str, str, str, str, str, str]] = []
    non_allowlisted_rows: list[tuple[str, str, str, str, str, str]] = []

    def add_row(token: str, kind: str, bmc_value: str, lec_value: str) -> None:
        case_id, fingerprint = (token.rsplit("::", 1) + [""])[:2]
        allowlisted = any(
            is_allowlisted(candidate, allow_exact, allow_prefix, allow_regex)
            for candidate in (
                f"{token}::{kind}",
                token,
                case_id,
                fingerprint,
            )
        )
        row = (
            case_id,
            fingerprint,
            kind,
            bmc_value,
            lec_value,
            "1" if allowlisted else "0",
        )
        parity_rows.append(row)
        if not allowlisted:
            non_allowlisted_rows.append(row)

    bmc_tokens = set(bmc.keys())
    lec_tokens = set(lec.keys())

    for token in sorted(bmc_tokens - lec_tokens):
        add_row(token, "missing_in_lec", "present", "absent")
    for token in sorted(lec_tokens - bmc_tokens):
        add_row(token, "missing_in_bmc", "absent", "present")

    if args.out_parity_tsv:
        emit_parity_tsv(Path(args.out_parity_tsv).resolve(), parity_rows)

    if non_allowlisted_rows:
        sample = ", ".join(
            f"{row[0]}::{row[1]}:{row[2]}" for row in non_allowlisted_rows[:6]
        )
        if len(non_allowlisted_rows) > 6:
            sample += ", ..."
        message = (
            "opentitan connectivity contract-fingerprint parity mismatches detected: "
            f"rows={len(non_allowlisted_rows)} sample=[{sample}] "
            f"bmc={bmc_path} lec={lec_path}"
        )
        if args.fail_on_mismatch:
            print(message, file=sys.stderr)
            raise SystemExit(1)
        print(f"warning: {message}", file=sys.stderr)
        return

    print(
        "opentitan connectivity contract-fingerprint parity check passed: "
        f"tokens_bmc={len(bmc)} tokens_lec={len(lec)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

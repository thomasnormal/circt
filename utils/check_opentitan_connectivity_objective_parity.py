#!/usr/bin/env python3
"""Check objective-level OpenTitan connectivity status parity between BMC and LEC.

This checker compares normalized per-objective statuses from connectivity lane
artifacts. It supports case objectives and optional cover objectives.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


STATUS_MAP = {
    "PASS": "pass",
    "FAIL": "fail",
    "XFAIL": "xfail",
    "XPASS": "xpass",
    "SKIP": "skip",
    "UNKNOWN": "error",
    "TIMEOUT": "error",
    "ERROR": "error",
}


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bmc-case-results",
        required=True,
        help="Connectivity BMC case-results TSV.",
    )
    parser.add_argument(
        "--lec-case-results",
        required=True,
        help="Connectivity LEC case-results TSV.",
    )
    parser.add_argument(
        "--bmc-cover-results",
        default="",
        help="Optional connectivity BMC cover-results TSV.",
    )
    parser.add_argument(
        "--lec-cover-results",
        default="",
        help="Optional connectivity LEC cover-results TSV.",
    )
    parser.add_argument(
        "--allowlist-file",
        default="",
        help=(
            "Optional allowlist file. Each non-comment line is exact:<token>, "
            "prefix:<prefix>, regex:<pattern>, or bare exact. Supported tokens are "
            "<objective_id> and <objective_id>::<kind>."
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
    parser.add_argument(
        "--include-missing-objectives",
        action="store_true",
        help=(
            "Also report objectives present in one lane and missing in the other. "
            "Default compares only shared objective IDs."
        ),
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


def normalize_status(raw: str) -> str:
    token = raw.strip().upper()
    if token in STATUS_MAP:
        return STATUS_MAP[token]
    return "error"


def read_objective_rows(path: Path, lane_name: str, kind: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            fail(
                f"{lane_name} {kind} results malformed row {line_no}: "
                f"expected >=2 columns"
            )
        status = normalize_status(parts[0])
        objective_payload = parts[1].strip()
        if not objective_payload:
            continue
        objective_id = f"{kind}::{objective_payload}"
        if objective_id in out:
            fail(
                f"duplicate objective_id '{objective_id}' in {path} row {line_no}"
            )
        out[objective_id] = status
    return out


def emit_parity_tsv(path: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["objective_id", "kind", "bmc", "lec", "allowlisted"])
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    bmc_case_path = Path(args.bmc_case_results).resolve()
    lec_case_path = Path(args.lec_case_results).resolve()
    bmc_cover_path = Path(args.bmc_cover_results).resolve() if args.bmc_cover_results else None
    lec_cover_path = Path(args.lec_cover_results).resolve() if args.lec_cover_results else None

    if not bmc_case_path.is_file():
        fail(f"connectivity BMC case-results file not found: {bmc_case_path}")
    if not lec_case_path.is_file():
        fail(f"connectivity LEC case-results file not found: {lec_case_path}")

    allow_exact: set[str] = set()
    allow_prefix: list[str] = []
    allow_regex: list[re.Pattern[str]] = []
    if args.allowlist_file:
        allow_exact, allow_prefix, allow_regex = load_allowlist(
            Path(args.allowlist_file).resolve()
        )

    bmc = read_objective_rows(bmc_case_path, "connectivity BMC", "case")
    lec = read_objective_rows(lec_case_path, "connectivity LEC", "case")
    if bmc_cover_path is not None:
        bmc.update(read_objective_rows(bmc_cover_path, "connectivity BMC", "cover"))
    if lec_cover_path is not None:
        lec.update(read_objective_rows(lec_cover_path, "connectivity LEC", "cover"))

    parity_rows: list[tuple[str, str, str, str, str]] = []
    non_allowlisted_rows: list[tuple[str, str, str, str, str]] = []

    def add_row(objective_id: str, kind: str, bmc_value: str, lec_value: str) -> None:
        tokens = (f"{objective_id}::{kind}", objective_id)
        allowlisted = any(
            is_allowlisted(token, allow_exact, allow_prefix, allow_regex)
            for token in tokens
        )
        row = (objective_id, kind, bmc_value, lec_value, "1" if allowlisted else "0")
        parity_rows.append(row)
        if not allowlisted:
            non_allowlisted_rows.append(row)

    bmc_ids = set(bmc.keys())
    lec_ids = set(lec.keys())

    if args.include_missing_objectives:
        for objective_id in sorted(bmc_ids - lec_ids):
            add_row(objective_id, "missing_in_lec", "present", "absent")
        for objective_id in sorted(lec_ids - bmc_ids):
            add_row(objective_id, "missing_in_bmc", "absent", "present")

    for objective_id in sorted(bmc_ids.intersection(lec_ids)):
        bmc_status = bmc[objective_id]
        lec_status = lec[objective_id]
        if bmc_status != lec_status:
            add_row(objective_id, "status", bmc_status, lec_status)

    if args.out_parity_tsv:
        emit_parity_tsv(Path(args.out_parity_tsv).resolve(), parity_rows)

    if non_allowlisted_rows:
        sample = ", ".join(
            f"{objective_id}:{kind}"
            for objective_id, kind, _, _, _ in non_allowlisted_rows[:6]
        )
        if len(non_allowlisted_rows) > 6:
            sample += ", ..."
        message = (
            "opentitan connectivity objective parity mismatches detected: "
            f"rows={len(non_allowlisted_rows)} sample=[{sample}] "
            f"bmc_case={bmc_case_path} lec_case={lec_case_path}"
        )
        if args.fail_on_mismatch:
            print(message, file=sys.stderr)
            raise SystemExit(1)
        print(f"warning: {message}", file=sys.stderr)
        return

    print(
        "opentitan connectivity objective parity check passed: "
        f"objectives_bmc={len(bmc)} objectives_lec={len(lec)} "
        f"shared={len(bmc_ids.intersection(lec_ids))}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

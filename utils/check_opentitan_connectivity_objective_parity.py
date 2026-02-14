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
from dataclasses import dataclass
from pathlib import Path


CASE_STATUS_MAP = {
    "PASS": "pass",
    "PROVEN": "pass",
    "FAIL": "fail",
    "FAILING": "fail",
    "XFAIL": "xfail",
    "XPASS": "xpass",
    "VACUOUS": "vacuous",
    "UNREACHABLE": "unreachable",
    "SKIP": "skip",
    "UNKNOWN": "unknown",
    "TIMEOUT": "timeout",
    "ERROR": "error",
}

COVER_STATUS_MAP = {
    "COVERED": "covered",
    "UNREACHABLE": "unreachable",
    "SKIP": "skip",
    "UNKNOWN": "unknown",
    "TIMEOUT": "timeout",
    "ERROR": "error",
}

MISSING_POLICY_VALUES = ("ignore", "case", "all")


@dataclass(frozen=True)
class ObjectiveEntry:
    objective_id: str
    objective_class: str
    objective_key: str
    rule_id: str
    status: str
    case_path: str


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
            "<objective_id>, <objective_id>::<kind>, and <rule_id>."
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
            "Deprecated alias for --missing-objective-policy=all. "
            "Include objective IDs present in one lane and missing in the other."
        ),
    )
    parser.add_argument(
        "--missing-objective-policy",
        choices=MISSING_POLICY_VALUES,
        default="ignore",
        help=(
            "Policy for objectives present in one lane and missing in the other: "
            "ignore (default), case (only case objectives), all (case+cover)."
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


def normalize_status(raw: str, objective_class: str) -> str:
    token = raw.strip().upper()
    if objective_class == "cover":
        if token in COVER_STATUS_MAP:
            return COVER_STATUS_MAP[token]
        return "error"
    if token in CASE_STATUS_MAP:
        return CASE_STATUS_MAP[token]
    return "error"


def parse_rule_id(objective_key: str) -> str:
    prefix = "connectivity::"
    if objective_key.startswith(prefix):
        return objective_key[len(prefix) :]
    return objective_key


def read_objective_rows(
    path: Path, lane_name: str, objective_class: str
) -> dict[str, ObjectiveEntry]:
    out: dict[str, ObjectiveEntry] = {}
    if not path.is_file():
        return out
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            fail(
                f"{lane_name} {objective_class} results malformed row {line_no}: "
                f"expected >=2 columns"
            )
        status = normalize_status(parts[0], objective_class)
        objective_key = parts[1].strip()
        if not objective_key:
            continue
        objective_id = f"{objective_class}::{objective_key}"
        case_path = parts[2].strip() if len(parts) >= 3 else ""
        if objective_id in out:
            fail(
                f"duplicate objective_id '{objective_id}' in {path} row {line_no}"
            )
        out[objective_id] = ObjectiveEntry(
            objective_id=objective_id,
            objective_class=objective_class,
            objective_key=objective_key,
            rule_id=parse_rule_id(objective_key),
            status=status,
            case_path=case_path,
        )
    return out


def emit_parity_tsv(
    path: Path, rows: list[tuple[str, str, str, str, str, str, str, str, str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "objective_id",
                "objective_class",
                "objective_key",
                "rule_id",
                "kind",
                "bmc",
                "lec",
                "bmc_case_path",
                "lec_case_path",
                "allowlisted",
            ]
        )
        for row in rows:
            writer.writerow(row)


def missing_policy_for_args(args: argparse.Namespace) -> str:
    policy = args.missing_objective_policy
    if args.include_missing_objectives and policy == "ignore":
        return "all"
    return policy


def main() -> None:
    args = parse_args()
    bmc_case_path = Path(args.bmc_case_results).resolve()
    lec_case_path = Path(args.lec_case_results).resolve()
    bmc_cover_path = (
        Path(args.bmc_cover_results).resolve() if args.bmc_cover_results else None
    )
    lec_cover_path = (
        Path(args.lec_cover_results).resolve() if args.lec_cover_results else None
    )
    missing_policy = missing_policy_for_args(args)

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

    parity_rows: list[tuple[str, str, str, str, str, str, str, str, str, str]] = []
    non_allowlisted_rows: list[
        tuple[str, str, str, str, str, str, str, str, str, str]
    ] = []

    def add_row(
        entry: ObjectiveEntry, kind: str, bmc_value: str, lec_value: str, lec_path: str
    ) -> None:
        tokens = (
            f"{entry.objective_id}::{kind}",
            entry.objective_id,
            entry.rule_id,
        )
        allowlisted = any(
            is_allowlisted(token, allow_exact, allow_prefix, allow_regex)
            for token in tokens
        )
        row = (
            entry.objective_id,
            entry.objective_class,
            entry.objective_key,
            entry.rule_id,
            kind,
            bmc_value,
            lec_value,
            entry.case_path,
            lec_path,
            "1" if allowlisted else "0",
        )
        parity_rows.append(row)
        if not allowlisted:
            non_allowlisted_rows.append(row)

    def include_missing_entry(entry: ObjectiveEntry) -> bool:
        if missing_policy == "ignore":
            return False
        if missing_policy == "case":
            return entry.objective_class == "case"
        return True

    def mismatch_kind(entry: ObjectiveEntry) -> str:
        if entry.objective_class == "cover":
            return "cover_status"
        return "case_status"

    bmc_ids = set(bmc.keys())
    lec_ids = set(lec.keys())

    for objective_id in sorted(bmc_ids - lec_ids):
        entry = bmc[objective_id]
        if include_missing_entry(entry):
            add_row(entry, "missing_in_lec", "present", "absent", "")
    for objective_id in sorted(lec_ids - bmc_ids):
        entry = lec[objective_id]
        if include_missing_entry(entry):
            mirror_entry = ObjectiveEntry(
                objective_id=entry.objective_id,
                objective_class=entry.objective_class,
                objective_key=entry.objective_key,
                rule_id=entry.rule_id,
                status=entry.status,
                case_path="",
            )
            add_row(mirror_entry, "missing_in_bmc", "absent", "present", entry.case_path)

    for objective_id in sorted(bmc_ids.intersection(lec_ids)):
        bmc_entry = bmc[objective_id]
        lec_entry = lec[objective_id]
        if bmc_entry.status != lec_entry.status:
            add_row(
                bmc_entry,
                mismatch_kind(bmc_entry),
                bmc_entry.status,
                lec_entry.status,
                lec_entry.case_path,
            )

    if args.out_parity_tsv:
        emit_parity_tsv(Path(args.out_parity_tsv).resolve(), parity_rows)

    if non_allowlisted_rows:
        sample = ", ".join(
            f"{row[0]}:{row[4]}" for row in non_allowlisted_rows[:6]
        )
        if len(non_allowlisted_rows) > 6:
            sample += ", ..."
        message = (
            "opentitan connectivity objective parity mismatches detected: "
            f"rows={len(non_allowlisted_rows)} sample=[{sample}] "
            f"bmc_case={bmc_case_path} lec_case={lec_case_path} "
            f"missing_policy={missing_policy}"
        )
        if args.fail_on_mismatch:
            print(message, file=sys.stderr)
            raise SystemExit(1)
        print(f"warning: {message}", file=sys.stderr)
        return

    print(
        "opentitan connectivity objective parity check passed: "
        f"objectives_bmc={len(bmc)} objectives_lec={len(lec)} "
        f"shared={len(bmc_ids.intersection(lec_ids))} "
        f"missing_policy={missing_policy}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

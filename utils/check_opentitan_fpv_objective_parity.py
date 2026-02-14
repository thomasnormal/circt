#!/usr/bin/env python3
"""Check objective-level OpenTitan FPV parity between BMC and LEC evidence.

This checker compares normalized per-objective statuses from FPV assertion and
cover evidence artifacts. It is intentionally lane-agnostic and can compare any
two objective evidence sets tagged as BMC/LEC for parity governance.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ASSERTION_STATUS_MAP = {
    "PASS": "proven",
    "PROVEN": "proven",
    "EQ": "proven",
    "FAIL": "failing",
    "FAILING": "failing",
    "NEQ": "failing",
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

MISSING_POLICY_VALUES = ("ignore", "assertion", "all")


@dataclass(frozen=True)
class ObjectiveEntry:
    objective_id: str
    objective_class: str
    target_name: str
    case_id: str
    objective_key: str
    status: str
    case_path: str


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bmc-assertion-results",
        required=True,
        help="OpenTitan FPV BMC assertion-results TSV.",
    )
    parser.add_argument(
        "--lec-assertion-results",
        required=True,
        help="OpenTitan FPV LEC assertion-results TSV.",
    )
    parser.add_argument(
        "--bmc-cover-results",
        default="",
        help="Optional OpenTitan FPV BMC cover-results TSV.",
    )
    parser.add_argument(
        "--lec-cover-results",
        default="",
        help="Optional OpenTitan FPV LEC cover-results TSV.",
    )
    parser.add_argument(
        "--allowlist-file",
        default="",
        help=(
            "Optional allowlist file. Each non-comment line is exact:<token>, "
            "prefix:<prefix>, regex:<pattern>, or bare exact. Supported tokens "
            "are <objective_id>, <objective_id>::<kind>, <case_id>, and "
            "<target_name>."
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
            "Deprecated alias for --missing-objective-policy=all. Include "
            "objectives present in one lane and missing in the other."
        ),
    )
    parser.add_argument(
        "--missing-objective-policy",
        choices=MISSING_POLICY_VALUES,
        default="ignore",
        help=(
            "Policy for objectives present in one lane and missing in the other: "
            "ignore (default), assertion (assertion objectives only), all "
            "(assertion+cover objectives)."
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
    if token in ASSERTION_STATUS_MAP:
        return ASSERTION_STATUS_MAP[token]
    return "error"


def parse_target_name(case_id: str) -> str:
    token = case_id.strip()
    if "::" in token:
        return token.split("::", 1)[0].strip()
    return token


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
        if len(parts) < 4:
            fail(
                f"{lane_name} {objective_class} results malformed row {line_no}: "
                "expected >=4 TSV columns"
            )
        status = normalize_status(parts[0], objective_class)
        case_id = parts[1].strip()
        case_path = parts[2].strip()
        objective_leaf = parts[3].strip()
        if not case_id or not objective_leaf:
            continue
        objective_key = f"{case_id}::{objective_leaf}"
        objective_id = f"{objective_class}::{objective_key}"
        if objective_id in out:
            fail(f"duplicate objective_id '{objective_id}' in {path} row {line_no}")
        out[objective_id] = ObjectiveEntry(
            objective_id=objective_id,
            objective_class=objective_class,
            target_name=parse_target_name(case_id),
            case_id=case_id,
            objective_key=objective_key,
            status=status,
            case_path=case_path,
        )
    return out


def emit_parity_tsv(
    path: Path,
    rows: list[tuple[str, str, str, str, str, str, str, str, str, str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "objective_id",
                "objective_class",
                "target_name",
                "case_id",
                "objective_key",
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
    bmc_assertion_path = Path(args.bmc_assertion_results).resolve()
    lec_assertion_path = Path(args.lec_assertion_results).resolve()
    bmc_cover_path = Path(args.bmc_cover_results).resolve() if args.bmc_cover_results else None
    lec_cover_path = Path(args.lec_cover_results).resolve() if args.lec_cover_results else None
    missing_policy = missing_policy_for_args(args)

    if not bmc_assertion_path.is_file():
        fail(f"OpenTitan FPV BMC assertion-results file not found: {bmc_assertion_path}")
    if not lec_assertion_path.is_file():
        fail(f"OpenTitan FPV LEC assertion-results file not found: {lec_assertion_path}")

    allow_exact: set[str] = set()
    allow_prefix: list[str] = []
    allow_regex: list[re.Pattern[str]] = []
    if args.allowlist_file:
        allow_exact, allow_prefix, allow_regex = load_allowlist(
            Path(args.allowlist_file).resolve()
        )

    bmc = read_objective_rows(bmc_assertion_path, "FPV BMC", "assertion")
    lec = read_objective_rows(lec_assertion_path, "FPV LEC", "assertion")
    if bmc_cover_path is not None:
        bmc.update(read_objective_rows(bmc_cover_path, "FPV BMC", "cover"))
    if lec_cover_path is not None:
        lec.update(read_objective_rows(lec_cover_path, "FPV LEC", "cover"))

    parity_rows: list[tuple[str, str, str, str, str, str, str, str, str, str, str]] = []
    non_allowlisted_rows: list[
        tuple[str, str, str, str, str, str, str, str, str, str, str]
    ] = []

    def add_row(
        entry: ObjectiveEntry, kind: str, bmc_value: str, lec_value: str, lec_path: str
    ) -> None:
        tokens = (
            f"{entry.objective_id}::{kind}",
            entry.objective_id,
            entry.case_id,
            entry.target_name,
        )
        allowlisted = any(
            is_allowlisted(token, allow_exact, allow_prefix, allow_regex)
            for token in tokens
        )
        row = (
            entry.objective_id,
            entry.objective_class,
            entry.target_name,
            entry.case_id,
            entry.objective_key,
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
        if missing_policy == "assertion":
            return entry.objective_class == "assertion"
        return True

    def mismatch_kind(entry: ObjectiveEntry) -> str:
        if entry.objective_class == "cover":
            return "cover_status"
        return "assertion_status"

    bmc_ids = set(bmc.keys())
    lec_ids = set(lec.keys())

    for objective_id in sorted(bmc_ids - lec_ids):
        entry = bmc[objective_id]
        if include_missing_entry(entry):
            add_row(entry, "missing_in_lec", "present", "absent", "")

    for objective_id in sorted(lec_ids - bmc_ids):
        entry = lec[objective_id]
        if include_missing_entry(entry):
            add_row(entry, "missing_in_bmc", "absent", "present", entry.case_path)

    for objective_id in sorted(bmc_ids & lec_ids):
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
        print("opentitan fpv objective parity mismatches detected", file=sys.stderr)
        sample = ", ".join(
            f"{row[0]}:{row[5]}" for row in non_allowlisted_rows[:5]
        )
        if sample:
            print(sample, file=sys.stderr)
        if args.fail_on_mismatch:
            raise SystemExit(1)
    else:
        print("opentitan fpv objective parity check passed", file=sys.stderr)


if __name__ == "__main__":
    main()

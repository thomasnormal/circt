#!/usr/bin/env python3
"""Check OpenTitan FPV BMC evidence parity against FPV summary rollups.

This checker recomputes per-target FPV rollup counters from raw evidence
artifacts (`case-results`, `assertion-results`, optional `cover-results`) and
compares them to the emitted FPV summary TSV.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


SUMMARY_FIELDS = (
    "total_assertions",
    "proven",
    "failing",
    "vacuous",
    "covered",
    "unreachable",
    "unknown",
    "error",
    "timeout",
    "skipped",
)


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-results",
        required=True,
        help="OpenTitan FPV BMC case-results TSV.",
    )
    parser.add_argument(
        "--assertion-results",
        required=True,
        help="OpenTitan FPV BMC assertion-results TSV.",
    )
    parser.add_argument(
        "--cover-results",
        default="",
        help="Optional OpenTitan FPV BMC cover-results TSV.",
    )
    parser.add_argument(
        "--fpv-summary",
        required=True,
        help="OpenTitan FPV BMC FPV-summary TSV.",
    )
    parser.add_argument(
        "--allowlist-file",
        default="",
        help=(
            "Optional allowlist file. Each non-comment line is exact:<token>, "
            "prefix:<prefix>, regex:<pattern>, or bare exact. Supported tokens are "
            "<target_name> and <target_name>::<kind>."
        ),
    )
    parser.add_argument(
        "--out-parity-tsv",
        default="",
        help="Optional output TSV path for parity mismatch rows.",
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


def init_counts() -> dict[str, int]:
    return {field: 0 for field in SUMMARY_FIELDS}


def count_assertion_like_status(counts: dict[str, int], status: str) -> None:
    counts["total_assertions"] += 1
    if status == "PROVEN":
        counts["proven"] += 1
    elif status == "FAILING":
        counts["failing"] += 1
    elif status == "VACUOUS":
        counts["vacuous"] += 1
    elif status == "COVERED":
        counts["covered"] += 1
    elif status == "UNREACHABLE":
        counts["unreachable"] += 1
    elif status == "UNKNOWN":
        counts["unknown"] += 1
    elif status == "TIMEOUT":
        counts["timeout"] += 1
    elif status == "SKIP":
        counts["skipped"] += 1
    else:
        counts["error"] += 1


def count_case_fallback_status(counts: dict[str, int], status: str) -> None:
    counts["total_assertions"] += 1
    if status == "PASS":
        counts["proven"] += 1
    elif status == "FAIL":
        counts["failing"] += 1
    elif status == "UNKNOWN":
        counts["unknown"] += 1
    elif status == "TIMEOUT":
        counts["timeout"] += 1
    elif status == "SKIP":
        counts["skipped"] += 1
    else:
        counts["error"] += 1


def parse_target(case_id: str) -> str:
    token = case_id.strip()
    if not token:
        return ""
    return token.split("::", 1)[0].strip()


def read_plain_rows(path: Path) -> list[tuple[str, ...]]:
    if not path.is_file():
        return []
    rows: list[tuple[str, ...]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(tuple(line.split("\t")))
    return rows


def compute_evidence_counts(
    case_rows: list[tuple[str, ...]],
    assertion_rows: list[tuple[str, ...]],
    cover_rows: list[tuple[str, ...]],
) -> dict[str, dict[str, str]]:
    by_target: dict[str, dict[str, int]] = {}

    def get_counts(target: str) -> dict[str, int]:
        return by_target.setdefault(target, init_counts())

    if assertion_rows or cover_rows:
        for row in [*assertion_rows, *cover_rows]:
            if len(row) < 2:
                continue
            status = (row[0] if row else "").strip().upper()
            target = parse_target(row[1] if len(row) > 1 else "")
            if not target:
                continue
            count_assertion_like_status(get_counts(target), status)
    else:
        for row in case_rows:
            if len(row) < 2:
                continue
            status = (row[0] if row else "").strip().upper()
            target = parse_target(row[1] if len(row) > 1 else "")
            if not target:
                continue
            count_case_fallback_status(get_counts(target), status)

    return {
        target: {kind: str(counts[kind]) for kind in SUMMARY_FIELDS}
        for target, counts in by_target.items()
    }


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        fail(f"OpenTitan FPV summary file not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"OpenTitan FPV summary missing header row: {path}")
        required = {"target_name", *SUMMARY_FIELDS}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            fail(
                f"OpenTitan FPV summary missing required columns {missing}: {path} "
                f"(found: {reader.fieldnames})"
            )
        out: dict[str, dict[str, str]] = {}
        for idx, row in enumerate(reader, start=2):
            target = (row.get("target_name") or "").strip()
            if not target:
                continue
            if target in out:
                fail(f"duplicate target_name '{target}' in {path} row {idx}")
            out[target] = {
                kind: (row.get(kind) or "").strip() for kind in SUMMARY_FIELDS
            }
    return out


def objective_class_for_kind(kind: str) -> str:
    if kind in ("proven", "failing", "vacuous"):
        return "assertion"
    if kind in ("covered", "unreachable"):
        return "cover"
    if kind == "total_assertions":
        return "aggregate"
    if kind in ("missing_in_evidence", "missing_in_summary"):
        return "target"
    return "mixed"


def emit_parity_tsv(
    path: Path, rows: list[tuple[str, str, str, str, str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["target_name", "objective_class", "kind", "evidence", "summary", "allowlisted"]
        )
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    case_path = Path(args.case_results).resolve()
    assertion_path = Path(args.assertion_results).resolve()
    cover_path = Path(args.cover_results).resolve() if args.cover_results else None
    summary_path = Path(args.fpv_summary).resolve()

    if not case_path.is_file():
        fail(f"OpenTitan FPV case-results file not found: {case_path}")
    if not assertion_path.is_file():
        fail(f"OpenTitan FPV assertion-results file not found: {assertion_path}")

    case_rows = read_plain_rows(case_path)
    assertion_rows = read_plain_rows(assertion_path)
    cover_rows = read_plain_rows(cover_path) if cover_path else []
    evidence = compute_evidence_counts(case_rows, assertion_rows, cover_rows)
    summary = read_summary(summary_path)

    allow_exact: set[str] = set()
    allow_prefix: list[str] = []
    allow_regex: list[re.Pattern[str]] = []
    if args.allowlist_file:
        allow_exact, allow_prefix, allow_regex = load_allowlist(
            Path(args.allowlist_file).resolve()
        )

    parity_rows: list[tuple[str, str, str, str, str, str]] = []
    non_allowlisted_rows: list[tuple[str, str, str, str, str, str]] = []

    def add_row(target: str, kind: str, evidence_value: str, summary_value: str) -> None:
        tokens = (f"{target}::{kind}", target)
        allowlisted = any(
            is_allowlisted(token, allow_exact, allow_prefix, allow_regex)
            for token in tokens
        )
        row = (
            target,
            objective_class_for_kind(kind),
            kind,
            evidence_value,
            summary_value,
            "1" if allowlisted else "0",
        )
        parity_rows.append(row)
        if not allowlisted:
            non_allowlisted_rows.append(row)

    evidence_targets = set(evidence.keys())
    summary_targets = set(summary.keys())
    for target in sorted(evidence_targets - summary_targets):
        add_row(target, "missing_in_summary", "present", "absent")
    for target in sorted(summary_targets - evidence_targets):
        add_row(target, "missing_in_evidence", "absent", "present")

    for target in sorted(evidence_targets.intersection(summary_targets)):
        evidence_row = evidence[target]
        summary_row = summary[target]
        for kind in SUMMARY_FIELDS:
            evidence_value = evidence_row.get(kind, "")
            summary_value = summary_row.get(kind, "")
            if evidence_value != summary_value:
                add_row(target, kind, evidence_value, summary_value)

    if args.out_parity_tsv:
        emit_parity_tsv(Path(args.out_parity_tsv).resolve(), parity_rows)

    if non_allowlisted_rows:
        sample = ", ".join(
            f"{target}:{kind}" for target, _, kind, _, _, _ in non_allowlisted_rows[:6]
        )
        if len(non_allowlisted_rows) > 6:
            sample += ", ..."
        message = (
            "opentitan fpv bmc evidence parity mismatches detected: "
            f"rows={len(non_allowlisted_rows)} sample=[{sample}] "
            f"case={case_path} assertions={assertion_path} summary={summary_path}"
        )
        if args.fail_on_mismatch:
            print(message, file=sys.stderr)
            raise SystemExit(1)
        print(f"warning: {message}", file=sys.stderr)
        return

    print(
        "opentitan fpv bmc evidence parity check passed: "
        f"targets_evidence={len(evidence)} targets_summary={len(summary)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

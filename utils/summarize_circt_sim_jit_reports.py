#!/usr/bin/env python3
"""Aggregate circt-sim JIT report deopt telemetry for burn-down triage."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JIT report JSON files or directories containing report JSONs.",
    )
    parser.add_argument(
        "--glob",
        default="*.json",
        help="Glob pattern used when scanning input directories (default: *.json).",
    )
    parser.add_argument(
        "--allowlist-file",
        default="",
        help=(
            "Optional allowlist file. Each non-comment line is exact:<token>, "
            "prefix:<prefix>, regex:<pattern>, or bare exact. Tokens may be "
            "reason:<reason>, reason_detail:<reason>:<detail>, "
            "process_id:<id>, process_name:<name>, "
            "process_reason:<name>:<reason>, or process_id_reason:<id>:<reason>."
        ),
    )
    parser.add_argument(
        "--fail-on-any-non-allowlisted-deopt",
        action="store_true",
        help="Exit non-zero when any non-allowlisted deopt process row exists.",
    )
    parser.add_argument(
        "--fail-on-reason",
        action="append",
        default=[],
        help="Exit non-zero when this deopt reason appears in non-allowlisted rows.",
    )
    parser.add_argument(
        "--fail-on-reason-detail",
        action="append",
        default=[],
        help=(
            "Exit non-zero when this reason/detail appears in non-allowlisted "
            "rows. Format: <reason>:<detail> (use '-' for empty detail)."
        ),
    )
    parser.add_argument(
        "--out-reason-tsv",
        default="",
        help="Optional TSV output path for ranked deopt reason counts.",
    )
    parser.add_argument(
        "--out-detail-tsv",
        default="",
        help="Optional TSV output path for ranked reason+detail counts.",
    )
    parser.add_argument(
        "--out-process-tsv",
        default="",
        help="Optional TSV output path for per-process deopt rows.",
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
                        f"invalid allowlist row {line_no}: "
                        f"bad regex '{payload}': {exc}"
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


def parse_reason_detail_spec(spec: str) -> tuple[str, str]:
    reason, sep, detail = spec.partition(":")
    reason = reason.strip()
    detail = detail.strip()
    if not sep or not reason:
        fail(
            "invalid --fail-on-reason-detail spec: "
            f"'{spec}' (expected <reason>:<detail>)"
        )
    return reason, detail


def detail_for_token(detail: str) -> str:
    return detail if detail else "-"


def collect_report_paths(inputs: Iterable[str], glob_pattern: str) -> list[Path]:
    reports: list[Path] = []
    for raw in inputs:
        path = Path(raw).resolve()
        if path.is_file():
            reports.append(path)
            continue
        if path.is_dir():
            for candidate in sorted(path.rglob(glob_pattern)):
                if candidate.is_file():
                    reports.append(candidate.resolve())
            continue
        fail(f"input path does not exist: {path}")

    deduped = sorted(set(reports))
    if not deduped:
        fail("no JIT report JSON files found")
    return deduped


def normalize_process_id(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return ""


def parse_report(path: Path) -> list[tuple[str, str, str, str, str]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON report {path}: {exc}")

    if not isinstance(report, dict):
        fail(f"invalid report root (expected object): {path}")

    jit = report.get("jit")
    if jit is None:
        return []
    if not isinstance(jit, dict):
        fail(f"invalid report field jit (expected object): {path}")

    deopt_rows = jit.get("jit_deopt_processes", [])
    if not isinstance(deopt_rows, list):
        fail(f"invalid report field jit.jit_deopt_processes (expected list): {path}")

    parsed_rows: list[tuple[str, str, str, str, str]] = []
    for row in deopt_rows:
        if not isinstance(row, dict):
            fail(
                "invalid report field jit.jit_deopt_processes[] "
                f"(expected object): {path}"
            )
        reason = str(row.get("reason", "unknown")).strip() or "unknown"
        detail = str(row.get("detail", "")).strip()
        process_id = normalize_process_id(row.get("process_id"))
        process_name = str(row.get("process_name", "")).strip()
        parsed_rows.append((str(path), process_id, process_name, reason, detail))

    return parsed_rows


def write_reason_tsv(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["rank", "reason", "count"])
        for rank, (reason, count) in enumerate(rows, start=1):
            writer.writerow([rank, reason, count])


def write_detail_tsv(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["rank", "reason", "detail", "count"])
        for rank, (reason, detail, count) in enumerate(rows, start=1):
            writer.writerow([rank, reason, detail if detail else "-", count])


def process_sort_key(process_id: str) -> tuple[int, object]:
    if process_id.isdigit():
        return (0, int(process_id))
    return (1, process_id)


def write_process_tsv(path: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row[0],
            process_sort_key(row[1]),
            row[3],
            row[4],
            row[2],
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["report", "process_id", "process_name", "reason", "detail"])
        for report, process_id, process_name, reason, detail in sorted_rows:
            writer.writerow([report, process_id, process_name, reason, detail])


def main() -> None:
    args = parse_args()

    allow_exact: set[str] = set()
    allow_prefix: list[str] = []
    allow_regex: list[re.Pattern[str]] = []
    if args.allowlist_file:
        allow_exact, allow_prefix, allow_regex = load_allowlist(
            Path(args.allowlist_file).resolve()
        )

    report_paths = collect_report_paths(args.inputs, args.glob)
    process_rows: list[tuple[str, str, str, str, str]] = []
    for report_path in report_paths:
        process_rows.extend(parse_report(report_path))

    allowlisted_flags: list[bool] = []
    reason_counts: Counter[str] = Counter()
    detail_counts: Counter[tuple[str, str]] = Counter()
    reason_counts_non_allowlisted: Counter[str] = Counter()
    detail_counts_non_allowlisted: Counter[tuple[str, str]] = Counter()
    for _, process_id, process_name, reason, detail in process_rows:
        reason_counts[reason] += 1
        detail_counts[(reason, detail)] += 1
        tokens = (
            f"reason:{reason}",
            f"reason_detail:{reason}:{detail_for_token(detail)}",
            f"process_id:{process_id}",
            f"process_name:{process_name}",
            f"process_reason:{process_name}:{reason}",
            f"process_id_reason:{process_id}:{reason}",
        )
        allowlisted = any(
            is_allowlisted(token, allow_exact, allow_prefix, allow_regex)
            for token in tokens
        )
        allowlisted_flags.append(allowlisted)
        if not allowlisted:
            reason_counts_non_allowlisted[reason] += 1
            detail_counts_non_allowlisted[(reason, detail)] += 1

    ranked_reasons = sorted(reason_counts.items(), key=lambda entry: (-entry[1], entry[0]))
    ranked_details = sorted(
        ((reason, detail, count) for (reason, detail), count in detail_counts.items()),
        key=lambda entry: (-entry[2], entry[0], entry[1]),
    )
    ranked_non_allowlisted_reasons = sorted(
        reason_counts_non_allowlisted.items(),
        key=lambda entry: (-entry[1], entry[0]),
    )
    ranked_non_allowlisted_details = sorted(
        (
            (reason, detail, count)
            for (reason, detail), count in detail_counts_non_allowlisted.items()
        ),
        key=lambda entry: (-entry[2], entry[0], entry[1]),
    )
    non_allowlisted_rows = len(process_rows) - sum(
        1 for is_allowed in allowlisted_flags if is_allowed
    )

    print(f"reports_scanned={len(report_paths)}")
    print(f"deopt_process_rows={len(process_rows)}")
    print(f"deopt_process_rows_non_allowlisted={non_allowlisted_rows}")
    print(f"unique_reasons={len(ranked_reasons)}")
    print(f"unique_reason_details={len(ranked_details)}")
    print(f"unique_reasons_non_allowlisted={len(ranked_non_allowlisted_reasons)}")
    print(
        f"unique_reason_details_non_allowlisted="
        f"{len(ranked_non_allowlisted_details)}"
    )

    for rank, (reason, count) in enumerate(ranked_reasons, start=1):
        print(f"top_reason[{rank}] reason={reason} count={count}")

    for rank, (reason, detail, count) in enumerate(ranked_details, start=1):
        detail_label = detail if detail else "-"
        print(
            f"top_reason_detail[{rank}] reason={reason} "
            f"detail={detail_label} count={count}"
        )

    for rank, (reason, count) in enumerate(ranked_non_allowlisted_reasons, start=1):
        print(f"top_non_allowlisted_reason[{rank}] reason={reason} count={count}")

    for rank, (reason, detail, count) in enumerate(
        ranked_non_allowlisted_details, start=1
    ):
        detail_label = detail_for_token(detail)
        print(
            f"top_non_allowlisted_reason_detail[{rank}] reason={reason} "
            f"detail={detail_label} count={count}"
        )

    if args.out_reason_tsv:
        write_reason_tsv(Path(args.out_reason_tsv).resolve(), ranked_reasons)
    if args.out_detail_tsv:
        write_detail_tsv(Path(args.out_detail_tsv).resolve(), ranked_details)
    if args.out_process_tsv:
        write_process_tsv(Path(args.out_process_tsv).resolve(), process_rows)

    failures: list[str] = []
    if args.fail_on_any_non_allowlisted_deopt and non_allowlisted_rows > 0:
        failures.append(
            "non-allowlisted deopts present: "
            f"rows={non_allowlisted_rows}"
        )
    for reason in args.fail_on_reason:
        reason_key = reason.strip()
        if not reason_key:
            fail("invalid --fail-on-reason value: empty")
        count = reason_counts_non_allowlisted.get(reason_key, 0)
        if count:
            failures.append(
                f"blocked reason matched: reason={reason_key} count={count}"
            )
    for spec in args.fail_on_reason_detail:
        reason_key, detail_key = parse_reason_detail_spec(spec)
        normalized_detail = "" if detail_key == "-" else detail_key
        count = detail_counts_non_allowlisted.get((reason_key, normalized_detail), 0)
        if count:
            failures.append(
                "blocked reason/detail matched: "
                f"reason={reason_key} detail={detail_for_token(normalized_detail)} "
                f"count={count}"
            )
    if failures:
        sys.stdout.flush()
        for msg in failures:
            print(msg, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check drift between OpenTitan compile-contract snapshots.

This compares the TSV artifact produced by
`utils/resolve_opentitan_formal_compile_contracts.py` across two runs and
reports target-level drift with deterministic diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_MARKER = "#opentitan_compile_contract_schema_version=1"


@dataclass(frozen=True)
class ContractRow:
    target_name: str
    task: str
    task_profile: str
    task_known: str
    stopat_mode: str
    blackbox_policy: str
    task_policy_fingerprint: str
    setup_status: str
    contract_fingerprint: str
    file_count: str
    include_dir_count: str
    define_count: str
    stopats_fingerprint: str


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline contracts TSV")
    parser.add_argument("--current", required=True, help="Current contracts TSV")
    parser.add_argument(
        "--allowlist-file",
        default="",
        help=(
            "Optional target-name allowlist file. Each non-comment line is "
            "exact:<name>, prefix:<prefix>, regex:<pattern>, or bare exact."
        ),
    )
    parser.add_argument(
        "--out-drift-tsv",
        default="",
        help="Optional output TSV path for drift rows.",
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


def read_contract_rows(path: Path) -> dict[str, ContractRow]:
    if not path.is_file():
        fail(f"contracts file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        fail(f"contracts file is empty: {path}")
    if lines[0].strip() != SCHEMA_MARKER:
        fail(
            f"missing/invalid schema marker in {path}: expected '{SCHEMA_MARKER}'"
        )
    body = lines[1:]
    if not body:
        fail(f"contracts file missing header row: {path}")

    reader = csv.DictReader(body, delimiter="\t")
    if reader.fieldnames is None:
        fail(f"contracts file missing header row: {path}")
    required = {
        "target_name",
        "task",
        "task_profile",
        "task_known",
        "stopat_mode",
        "blackbox_policy",
        "task_policy_fingerprint",
        "setup_status",
        "contract_fingerprint",
        "file_count",
        "include_dir_count",
        "define_count",
    }
    missing = sorted(required.difference(reader.fieldnames))
    if missing:
        fail(
            f"contracts file missing required columns {missing}: {path} "
            f"(found: {reader.fieldnames})"
        )

    out: dict[str, ContractRow] = {}
    for idx, row in enumerate(reader, start=3):
        target_name = (row.get("target_name") or "").strip()
        if not target_name:
            continue
        if target_name in out:
            fail(f"duplicate target_name '{target_name}' in {path} row {idx}")
        out[target_name] = ContractRow(
            target_name=target_name,
            task=(row.get("task") or "").strip(),
            task_profile=(row.get("task_profile") or "").strip(),
            task_known=(row.get("task_known") or "").strip(),
            stopat_mode=(row.get("stopat_mode") or "").strip(),
            blackbox_policy=(row.get("blackbox_policy") or "").strip(),
            task_policy_fingerprint=(row.get("task_policy_fingerprint") or "").strip(),
            setup_status=(row.get("setup_status") or "").strip(),
            contract_fingerprint=(row.get("contract_fingerprint") or "").strip(),
            file_count=(row.get("file_count") or "").strip(),
            include_dir_count=(row.get("include_dir_count") or "").strip(),
            define_count=(row.get("define_count") or "").strip(),
            stopats_fingerprint=(row.get("stopats_fingerprint") or "").strip(),
        )
    return out


def emit_drift_tsv(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["target_name", "kind", "baseline", "current"])
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline).resolve()
    current_path = Path(args.current).resolve()
    baseline = read_contract_rows(baseline_path)
    current = read_contract_rows(current_path)

    allow_exact: set[str] = set()
    allow_prefix: list[str] = []
    allow_regex: list[re.Pattern[str]] = []
    if args.allowlist_file:
        allow_exact, allow_prefix, allow_regex = load_allowlist(
            Path(args.allowlist_file).resolve()
        )

    drift_rows: list[tuple[str, str, str, str]] = []

    baseline_targets = set(baseline.keys())
    current_targets = set(current.keys())

    for target in sorted(baseline_targets - current_targets):
        if is_allowlisted(target, allow_exact, allow_prefix, allow_regex):
            continue
        drift_rows.append((target, "missing_in_current", "present", "absent"))

    for target in sorted(current_targets - baseline_targets):
        if is_allowlisted(target, allow_exact, allow_prefix, allow_regex):
            continue
        drift_rows.append((target, "new_in_current", "absent", "present"))

    for target in sorted(baseline_targets.intersection(current_targets)):
        if is_allowlisted(target, allow_exact, allow_prefix, allow_regex):
            continue
        b = baseline[target]
        c = current[target]
        if b.task != c.task:
            drift_rows.append((target, "task", b.task, c.task))
        if b.task_profile != c.task_profile:
            drift_rows.append((target, "task_profile", b.task_profile, c.task_profile))
        if b.task_known != c.task_known:
            drift_rows.append((target, "task_known", b.task_known, c.task_known))
        if b.stopat_mode != c.stopat_mode:
            drift_rows.append((target, "stopat_mode", b.stopat_mode, c.stopat_mode))
        if b.blackbox_policy != c.blackbox_policy:
            drift_rows.append(
                (target, "blackbox_policy", b.blackbox_policy, c.blackbox_policy)
            )
        if b.task_policy_fingerprint != c.task_policy_fingerprint:
            drift_rows.append(
                (
                    target,
                    "task_policy_fingerprint",
                    b.task_policy_fingerprint,
                    c.task_policy_fingerprint,
                )
            )
        if b.setup_status != c.setup_status:
            drift_rows.append((target, "setup_status", b.setup_status, c.setup_status))
        if b.contract_fingerprint != c.contract_fingerprint:
            drift_rows.append(
                (
                    target,
                    "contract_fingerprint",
                    b.contract_fingerprint,
                    c.contract_fingerprint,
                )
            )
        if b.file_count != c.file_count:
            drift_rows.append((target, "file_count", b.file_count, c.file_count))
        if b.include_dir_count != c.include_dir_count:
            drift_rows.append(
                (
                    target,
                    "include_dir_count",
                    b.include_dir_count,
                    c.include_dir_count,
                )
            )
        if b.define_count != c.define_count:
            drift_rows.append((target, "define_count", b.define_count, c.define_count))
        if b.stopats_fingerprint != c.stopats_fingerprint:
            drift_rows.append(
                (
                    target,
                    "stopats_fingerprint",
                    b.stopats_fingerprint,
                    c.stopats_fingerprint,
                )
            )

    if args.out_drift_tsv:
        emit_drift_tsv(Path(args.out_drift_tsv).resolve(), drift_rows)

    if drift_rows:
        sample = ", ".join(
            f"{target}:{kind}" for target, kind, _, _ in drift_rows[:6]
        )
        if len(drift_rows) > 6:
            sample += ", ..."
        print(
            (
                "opentitan compile contract drift detected: "
                f"rows={len(drift_rows)} sample=[{sample}] "
                f"baseline={baseline_path} current={current_path}"
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(
        (
            "opentitan compile contract drift check passed: "
            f"targets={len(current)} baseline={baseline_path} current={current_path}"
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

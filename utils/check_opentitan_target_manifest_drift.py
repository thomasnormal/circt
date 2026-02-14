#!/usr/bin/env python3
"""Check drift between OpenTitan FPV target-manifest snapshots.

This compares the TSV artifact produced by `utils/select_opentitan_formal_cfgs.py`
across two runs and reports target-level metadata drift with deterministic
diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TargetRow:
    target_name: str
    dut: str
    fusesoc_core: str
    task: str
    stopats: str
    flow: str
    sub_flow: str
    rel_path: str
    source_kind: str


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline manifest TSV")
    parser.add_argument("--current", required=True, help="Current manifest TSV")
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


def read_manifest_rows(path: Path) -> tuple[list[str], dict[str, TargetRow]]:
    if not path.is_file():
        fail(f"target manifest file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"target manifest missing header row: {path}")
        required = {
            "target_name",
            "dut",
            "fusesoc_core",
            "task",
            "stopats",
            "flow",
            "sub_flow",
            "rel_path",
            "source_kind",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            fail(
                f"target manifest missing required columns {missing}: {path} "
                f"(found: {reader.fieldnames})"
            )

        ordered_names: list[str] = []
        rows: dict[str, TargetRow] = {}
        for idx, row in enumerate(reader, start=2):
            target_name = (row.get("target_name") or "").strip()
            if not target_name:
                continue
            if target_name in rows:
                fail(f"duplicate target_name '{target_name}' in {path} row {idx}")
            ordered_names.append(target_name)
            rows[target_name] = TargetRow(
                target_name=target_name,
                dut=(row.get("dut") or "").strip(),
                fusesoc_core=(row.get("fusesoc_core") or "").strip(),
                task=(row.get("task") or "").strip(),
                stopats=(row.get("stopats") or "").strip(),
                flow=(row.get("flow") or "").strip(),
                sub_flow=(row.get("sub_flow") or "").strip(),
                rel_path=(row.get("rel_path") or "").strip(),
                source_kind=(row.get("source_kind") or "").strip(),
            )
    return ordered_names, rows


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
    baseline_order, baseline = read_manifest_rows(baseline_path)
    current_order, current = read_manifest_rows(current_path)

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

    max_order = max(len(baseline_order), len(current_order))
    for idx in range(max_order):
        b_name = baseline_order[idx] if idx < len(baseline_order) else ""
        c_name = current_order[idx] if idx < len(current_order) else ""
        if b_name == c_name:
            continue
        target = c_name or b_name or f"index:{idx}"
        if is_allowlisted(target, allow_exact, allow_prefix, allow_regex):
            continue
        drift_rows.append((target, "target_order", b_name, c_name))

    for target in sorted(baseline_targets.intersection(current_targets)):
        if is_allowlisted(target, allow_exact, allow_prefix, allow_regex):
            continue
        b = baseline[target]
        c = current[target]
        if b.dut != c.dut:
            drift_rows.append((target, "dut", b.dut, c.dut))
        if b.fusesoc_core != c.fusesoc_core:
            drift_rows.append((target, "fusesoc_core", b.fusesoc_core, c.fusesoc_core))
        if b.task != c.task:
            drift_rows.append((target, "task", b.task, c.task))
        if b.stopats != c.stopats:
            drift_rows.append((target, "stopats", b.stopats, c.stopats))
        if b.flow != c.flow:
            drift_rows.append((target, "flow", b.flow, c.flow))
        if b.sub_flow != c.sub_flow:
            drift_rows.append((target, "sub_flow", b.sub_flow, c.sub_flow))
        if b.rel_path != c.rel_path:
            drift_rows.append((target, "rel_path", b.rel_path, c.rel_path))
        if b.source_kind != c.source_kind:
            drift_rows.append((target, "source_kind", b.source_kind, c.source_kind))

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
                "opentitan fpv target-manifest drift detected: "
                f"rows={len(drift_rows)} sample=[{sample}] "
                f"baseline={baseline_path} current={current_path}"
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(
        (
            "opentitan fpv target-manifest drift check passed: "
            f"targets={len(current)} baseline={baseline_path} current={current_path}"
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

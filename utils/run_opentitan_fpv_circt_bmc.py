#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run OpenTitan FPV targets through generic circt-bmc pairwise runner.

This script consumes compile contracts emitted by
`resolve_opentitan_formal_compile_contracts.py`, expands target+toplevel cases,
and delegates execution to `run_pairwise_circt_bmc.py`.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


SCHEMA_MARKER = "#opentitan_compile_contract_schema_version=1"


@dataclass(frozen=True)
class ContractRow:
    target_name: str
    rel_path: str
    task_profile: str
    task_known: bool
    setup_status: str
    stopat_mode: str
    stopat_count: int
    stopats: tuple[str, ...]
    blackbox_modules: tuple[str, ...]
    toplevels: tuple[str, ...]
    files: tuple[str, ...]
    include_dirs: tuple[str, ...]
    defines: tuple[str, ...]


@dataclass(frozen=True)
class FPVSummaryRow:
    target_name: str
    total_assertions: str
    proven: str
    failing: str
    vacuous: str
    covered: str
    unreachable: str
    unknown: str
    error: str
    timeout: str
    skipped: str


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_semicolon_list(raw: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in raw.split(";") if token.strip())


def parse_toplevels(raw: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def parse_nonnegative_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        fail(f"invalid {name}: {raw}")
    if value < 0:
        fail(f"invalid {name}: {raw}")
    return value


def parse_bool(raw: str, name: str) -> bool:
    token = raw.strip().lower()
    if token in {"1", "true", "yes"}:
        return True
    if token in {"0", "false", "no"}:
        return False
    fail(f"invalid {name}: {raw}")


def parse_blackbox_modules(raw: str) -> tuple[str, ...]:
    text = raw.strip()
    if not text or text.lower() == "none":
        return ()
    modules: list[str] = []
    seen: set[str] = set()
    for token in text.split(","):
        module = token.strip()
        if not module:
            continue
        if not re.fullmatch(r"[^,\s]+", module):
            fail(f"invalid blackbox policy token: {module}")
        if module in seen:
            continue
        seen.add(module)
        modules.append(module)
    if not modules:
        return ()
    return tuple(modules)


def normalize_stopat_selector(raw: str) -> str:
    token = raw.strip()
    if not token:
        raise ValueError("empty stopat selector")
    if token.startswith("*"):
        token = token[1:].strip()
    parts = [part.strip() for part in token.split(".")]
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError(
            "unsupported stopat selector "
            f"'{raw}': expected 'inst[.inst...].port' or '*inst[.inst...].port'"
        )
    return ".".join(parts)


def read_compile_contracts(path: Path) -> list[ContractRow]:
    if not path.is_file():
        fail(f"compile-contracts file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        fail(f"compile-contracts file is empty: {path}")
    if lines[0].strip() != SCHEMA_MARKER:
        fail(
            f"missing/invalid schema marker in {path}: expected '{SCHEMA_MARKER}'"
        )
    body = lines[1:]
    if not body:
        fail(f"compile-contracts file missing header row: {path}")
    reader = csv.DictReader(body, delimiter="\t")
    if reader.fieldnames is None:
        fail(f"compile-contracts file missing header row: {path}")
    required = {
        "target_name",
        "task_profile",
        "task_known",
        "setup_status",
        "stopat_mode",
        "stopat_count",
        "stopats",
        "blackbox_policy",
        "toplevel",
        "files",
        "include_dirs",
        "defines",
    }
    missing = sorted(required.difference(reader.fieldnames))
    if missing:
        fail(
            f"compile-contracts file missing required columns {missing}: {path} "
            f"(found: {reader.fieldnames})"
        )
    out: list[ContractRow] = []
    for idx, row in enumerate(reader, start=3):
        target_name = (row.get("target_name") or "").strip()
        if not target_name:
            continue
        setup_status = (row.get("setup_status") or "").strip().lower()
        if not setup_status:
            fail(f"compile-contracts row {idx} missing setup_status in {path}")
        out.append(
            ContractRow(
                target_name=target_name,
                rel_path=(row.get("rel_path") or "").strip(),
                task_profile=(row.get("task_profile") or "").strip(),
                task_known=parse_bool((row.get("task_known") or "").strip(), "task_known"),
                setup_status=setup_status,
                stopat_mode=(row.get("stopat_mode") or "").strip().lower(),
                stopat_count=parse_nonnegative_int(
                    (row.get("stopat_count") or "0").strip(), "stopat_count"
                ),
                stopats=parse_semicolon_list((row.get("stopats") or "").strip()),
                blackbox_modules=parse_blackbox_modules(
                    (row.get("blackbox_policy") or "").strip()
                ),
                toplevels=parse_toplevels((row.get("toplevel") or "").strip()),
                files=parse_semicolon_list((row.get("files") or "").strip()),
                include_dirs=parse_semicolon_list(
                    (row.get("include_dirs") or "").strip()
                ),
                defines=parse_semicolon_list((row.get("defines") or "").strip()),
            )
        )
    return out


def summarize(rows: list[tuple[str, ...]]) -> tuple[int, int, int, int, int, int, int]:
    total = len(rows)
    passed = 0
    failed = 0
    xfailed = 0
    xpassed = 0
    errored = 0
    skipped = 0
    for row in rows:
        status = row[0].strip().upper() if row else ""
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        elif status == "XFAIL":
            xfailed += 1
        elif status == "XPASS":
            xpassed += 1
        elif status == "SKIP":
            skipped += 1
        else:
            errored += 1
    return total, passed, failed, xfailed, xpassed, errored, skipped


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


def write_fpv_summary(
    case_rows: list[tuple[str, ...]],
    assertion_rows: list[tuple[str, ...]],
    cover_rows: list[tuple[str, ...]],
    out_path: Path,
) -> None:
    # status, case_id, case_path, assertion_id, assertion_label, diag, reason
    # status, case_id, case_path, cover_id, cover_label, diag, reason
    # If no assertion/cover rows are available, fall back to case-level
    # accounting (one synthetic assertion per case).
    by_target: dict[str, dict[str, int]] = {}

    def init_counts(target: str) -> dict[str, int]:
        return by_target.setdefault(
            target,
            {
                "total_assertions": 0,
                "proven": 0,
                "failing": 0,
                "vacuous": 0,
                "covered": 0,
                "unreachable": 0,
                "unknown": 0,
                "error": 0,
                "timeout": 0,
                "skipped": 0,
            },
        )

    if assertion_rows or cover_rows:
        property_rows: list[tuple[str, ...]] = []
        property_rows.extend(assertion_rows)
        property_rows.extend(cover_rows)
        for row in property_rows:
            if len(row) < 2:
                continue
            status = row[0].strip().upper()
            case_id = row[1].strip()
            if not case_id:
                continue
            target_name = case_id.split("::", 1)[0]
            counts = init_counts(target_name)
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
    else:
        for row in case_rows:
            if len(row) < 2:
                continue
            status = row[0].strip().upper()
            case_id = row[1].strip()
            if not case_id:
                continue
            target_name = case_id.split("::", 1)[0]
            counts = init_counts(target_name)
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "target_name\ttotal_assertions\tproven\tfailing\tvacuous\tcovered\t"
            "unreachable\tunknown\terror\ttimeout\tskipped\n"
        )
        for target_name in sorted(by_target):
            c = by_target[target_name]
            handle.write(
                f"{target_name}\t{c['total_assertions']}\t{c['proven']}\t"
                f"{c['failing']}\t{c['vacuous']}\t{c['covered']}\t"
                f"{c['unreachable']}\t{c['unknown']}\t{c['error']}\t"
                f"{c['timeout']}\t{c['skipped']}\n"
            )


def read_fpv_summary(path: Path) -> dict[str, FPVSummaryRow]:
    if not path.is_file():
        fail(f"fpv summary file not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"fpv summary file missing header row: {path}")
        required = {
            "target_name",
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
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            fail(
                f"fpv summary file missing required columns {missing}: {path} "
                f"(found: {reader.fieldnames})"
            )

        out: dict[str, FPVSummaryRow] = {}
        for idx, row in enumerate(reader, start=2):
            target_name = (row.get("target_name") or "").strip()
            if not target_name:
                continue
            if target_name in out:
                fail(f"duplicate target_name '{target_name}' in {path} row {idx}")
            out[target_name] = FPVSummaryRow(
                target_name=target_name,
                total_assertions=(row.get("total_assertions") or "").strip(),
                proven=(row.get("proven") or "").strip(),
                failing=(row.get("failing") or "").strip(),
                vacuous=(row.get("vacuous") or "").strip(),
                covered=(row.get("covered") or "").strip(),
                unreachable=(row.get("unreachable") or "").strip(),
                unknown=(row.get("unknown") or "").strip(),
                error=(row.get("error") or "").strip(),
                timeout=(row.get("timeout") or "").strip(),
                skipped=(row.get("skipped") or "").strip(),
            )
    return out


def write_fpv_summary_drift(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["target_name", "kind", "baseline", "current"])
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenTitan FPV targets using compile contracts."
    )
    parser.add_argument(
        "--compile-contracts",
        required=True,
        help="OpenTitan FPV compile-contract TSV file",
    )
    parser.add_argument(
        "--target-filter",
        default="",
        help="Optional regex filter over compile-contract target names",
    )
    parser.add_argument(
        "--workdir",
        default="",
        help="Optional work directory (default: temp directory).",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep the work directory instead of deleting it.",
    )
    parser.add_argument(
        "--results-file",
        default=os.environ.get("OUT", ""),
        help="Optional TSV output path for per-target/per-top rows.",
    )
    parser.add_argument(
        "--drop-remark-cases-file",
        default=os.environ.get("BMC_DROP_REMARK_CASES_OUT", ""),
        help="Optional TSV output path for dropped-syntax case IDs.",
    )
    parser.add_argument(
        "--drop-remark-reasons-file",
        default=os.environ.get("BMC_DROP_REMARK_REASONS_OUT", ""),
        help="Optional TSV output path for dropped-syntax case+reason rows.",
    )
    parser.add_argument(
        "--timeout-reasons-file",
        default=os.environ.get("BMC_TIMEOUT_REASON_CASES_OUT", ""),
        help="Optional TSV output path for timeout reason rows.",
    )
    parser.add_argument(
        "--resolved-contracts-file",
        default=os.environ.get("BMC_RESOLVED_CONTRACTS_OUT", ""),
        help="Optional TSV output path for resolved per-case contract rows.",
    )
    parser.add_argument(
        "--assertion-results-file",
        default=os.environ.get("BMC_ASSERTION_RESULTS_OUT", ""),
        help="Optional TSV output path for per-assertion FPV BMC rows.",
    )
    parser.add_argument(
        "--cover-results-file",
        default=os.environ.get("BMC_COVER_RESULTS_OUT", ""),
        help="Optional TSV output path for per-cover FPV BMC rows.",
    )
    parser.add_argument(
        "--assertion-granular",
        action="store_true",
        default=os.environ.get("BMC_ASSERTION_GRANULAR", "0") == "1",
        help=(
            "Run BMC per assertion by delegating --assertion-granular to the "
            "pairwise runner (default: env BMC_ASSERTION_GRANULAR or off)."
        ),
    )
    parser.add_argument(
        "--assertion-granular-max",
        default=os.environ.get("BMC_ASSERTION_GRANULAR_MAX", "0"),
        help=(
            "Maximum assertions per case for --assertion-granular "
            "(0 means unlimited; default: env BMC_ASSERTION_GRANULAR_MAX or 0)."
        ),
    )
    parser.add_argument(
        "--cover-granular",
        action="store_true",
        default=os.environ.get("BMC_COVER_GRANULAR", "0") == "1",
        help=(
            "Run BMC per cover by delegating --cover-granular to the pairwise "
            "runner (default: env BMC_COVER_GRANULAR or off)."
        ),
    )
    parser.add_argument(
        "--fpv-summary-file",
        default=os.environ.get("BMC_FPV_SUMMARY_OUT", ""),
        help="Optional TSV output path for FPV-style assertion summary rows.",
    )
    parser.add_argument(
        "--fpv-summary-baseline-file",
        default=os.environ.get("BMC_FPV_SUMMARY_BASELINE_FILE", ""),
        help="Optional baseline FPV summary TSV file for drift checking.",
    )
    parser.add_argument(
        "--fpv-summary-drift-file",
        default=os.environ.get("BMC_FPV_SUMMARY_DRIFT_OUT", ""),
        help="Optional output TSV path for FPV summary drift rows.",
    )
    parser.add_argument(
        "--fpv-summary-drift-allowlist-file",
        default=os.environ.get("BMC_FPV_SUMMARY_DRIFT_ALLOWLIST_FILE", ""),
        help=(
            "Optional target-name allowlist file for FPV summary drift "
            "suppression. Each non-comment line is exact:<name>, "
            "prefix:<prefix>, regex:<pattern>, or bare exact."
        ),
    )
    parser.add_argument(
        "--fail-on-fpv-summary-drift",
        action="store_true",
        default=os.environ.get("BMC_FAIL_ON_FPV_SUMMARY_DRIFT", "0") == "1",
        help="Fail when FPV summary drift is detected vs baseline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contracts_path = Path(args.compile_contracts).resolve()
    contracts = read_compile_contracts(contracts_path)
    if not contracts:
        print("No OpenTitan FPV compile-contract rows found.", file=sys.stderr)
        return 1

    target_filter = args.target_filter.strip()
    target_re: re.Pattern[str] | None = None
    if target_filter:
        try:
            target_re = re.compile(target_filter)
        except re.error as exc:
            fail(f"invalid --target-filter: {target_filter} ({exc})")
    selected = [
        row
        for row in contracts
        if target_re is None or target_re.search(row.target_name)
    ]
    if not selected:
        print("No OpenTitan FPV compile-contract targets selected.", file=sys.stderr)
        return 1

    mode_label = os.environ.get("BMC_MODE_LABEL", "FPV_BMC").strip() or "FPV_BMC"
    bound = parse_nonnegative_int(os.environ.get("BOUND", "1"), "BOUND")
    if bound == 0:
        bound = 1
    ignore_asserts_until = parse_nonnegative_int(
        os.environ.get("IGNORE_ASSERTS_UNTIL", "0"), "IGNORE_ASSERTS_UNTIL"
    )
    assertion_granular_max = parse_nonnegative_int(
        args.assertion_granular_max, "--assertion-granular-max"
    )

    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        keep_workdir = True
    else:
        workdir = Path(tempfile.mkdtemp(prefix="opentitan-fpv-bmc-"))
        keep_workdir = args.keep_workdir

    pre_rows: list[tuple[str, ...]] = []
    grouped_case_lines: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}

    def add_contract_error(row: ContractRow, reason: str) -> None:
        case_id = row.target_name
        case_path = row.rel_path or row.target_name
        pre_rows.append(
            (
                "ERROR",
                case_id,
                case_path,
                "opentitan",
                mode_label,
                "CIRCT_BMC_ERROR",
                reason,
            )
        )

    for row in selected:
        if row.setup_status == "error":
            add_contract_error(row, "compile_contract_setup_error")
            continue
        if not row.task_known:
            add_contract_error(row, "compile_contract_unknown_task")
            continue
        if row.stopat_mode not in {"none", "task_defined"}:
            add_contract_error(row, "compile_contract_invalid_stopat_mode")
            continue
        if row.stopat_mode == "none" and row.stopat_count > 0:
            add_contract_error(row, "compile_contract_stopat_mode_none_with_stopats")
            continue
        if row.stopat_count != len(row.stopats):
            add_contract_error(row, "compile_contract_stopat_count_mismatch")
            continue
        normalized_stopats: tuple[str, ...] = ()
        if row.stopat_mode == "task_defined" and row.stopat_count > 0:
            try:
                normalized_stopats = tuple(
                    sorted(
                        {
                            normalize_stopat_selector(stopat)
                            for stopat in row.stopats
                        }
                    )
                )
            except ValueError:
                add_contract_error(row, "unsupported_stopat_selector")
                continue
        if not row.toplevels:
            add_contract_error(row, "compile_contract_missing_toplevel")
            continue
        if not row.files:
            add_contract_error(row, "compile_contract_missing_files")
            continue
        for top in row.toplevels:
            case_id = f"{row.target_name}::{top}"
            case_path = (
                f"{row.rel_path}/{top}" if row.rel_path else f"{row.target_name}/{top}"
            )
            blackbox_policy = (
                ",".join(row.blackbox_modules) if row.blackbox_modules else "none"
            )
            stopat_policy = ",".join(normalized_stopats) if normalized_stopats else "none"
            contract_source = (
                f"fpv_target:{row.target_name};"
                f"task_profile:{row.task_profile or 'unknown'};"
                f"stopat_mode:{row.stopat_mode or 'none'};"
                f"stopat_selectors:{stopat_policy};"
                f"blackbox_policy:{blackbox_policy}"
            )
            grouped_case_lines.setdefault(
                (normalized_stopats, row.blackbox_modules), []
            ).append(
                "\t".join(
                    [
                        case_id,
                        top,
                        ";".join(row.files),
                        ";".join(row.include_dirs),
                        case_path,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        contract_source,
                        ";".join(row.defines),
                    ]
                )
            )

    pairwise_runner = Path(__file__).resolve().with_name("run_pairwise_circt_bmc.py")
    if grouped_case_lines and not pairwise_runner.is_file():
        fail(f"missing pairwise runner: {pairwise_runner}")

    pairwise_result_files: list[Path] = []
    assertion_result_files: list[Path] = []
    cover_result_files: list[Path] = []
    drop_case_files: list[Path] = []
    drop_reason_files: list[Path] = []
    timeout_reason_files: list[Path] = []
    resolved_contract_files: list[Path] = []

    def merge_plain_files(sources: list[Path], dest: str) -> None:
        if not dest:
            return
        out_path = Path(dest)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as out_handle:
            for src in sources:
                if not src.exists():
                    continue
                content = src.read_text(encoding="utf-8")
                if not content:
                    continue
                out_handle.write(content)
                if not content.endswith("\n"):
                    out_handle.write("\n")

    def merge_resolved_contract_files(sources: list[Path], dest: str) -> None:
        if not dest:
            return
        out_path = Path(dest)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        marker_seen = False
        with out_path.open("w", encoding="utf-8") as out_handle:
            for src in sources:
                if not src.exists():
                    continue
                for line in src.read_text(encoding="utf-8").splitlines():
                    if not line:
                        continue
                    if line.startswith("#resolved_contract_schema_version="):
                        if marker_seen:
                            continue
                        marker_seen = True
                    out_handle.write(line + "\n")

    pairwise_rc = 0
    try:
        if grouped_case_lines:
            base_prepare_core_passes = os.environ.get(
                "BMC_PREPARE_CORE_PASSES",
                "--lower-lec-llvm --reconcile-unrealized-casts",
            ).strip()
            for group_index, group_key in enumerate(sorted(grouped_case_lines)):
                stopat_selectors, blackbox_modules = group_key
                group_cases = grouped_case_lines[group_key]
                group_cases_path = workdir / f"pairwise-cases-{group_index}.tsv"
                group_results_path = workdir / f"pairwise-results-{group_index}.tsv"
                group_workdir = workdir / f"pairwise-work-{group_index}"
                group_cases_path.write_text(
                    "\n".join(group_cases) + "\n", encoding="utf-8"
                )
                pairwise_result_files.append(group_results_path)

                cmd = [
                    sys.executable,
                    str(pairwise_runner),
                    "--cases-file",
                    str(group_cases_path),
                    "--suite-name",
                    "opentitan",
                    "--mode-label",
                    mode_label,
                    "--bound",
                    str(bound),
                    "--ignore-asserts-until",
                    str(ignore_asserts_until),
                    "--workdir",
                    str(group_workdir),
                    "--keep-workdir",
                    "--results-file",
                    str(group_results_path),
                ]

                if args.drop_remark_cases_file:
                    group_drop_cases = (
                        workdir / f"pairwise-drop-remark-cases-{group_index}.tsv"
                    )
                    drop_case_files.append(group_drop_cases)
                    cmd += ["--drop-remark-cases-file", str(group_drop_cases)]
                if args.drop_remark_reasons_file:
                    group_drop_reasons = (
                        workdir / f"pairwise-drop-remark-reasons-{group_index}.tsv"
                    )
                    drop_reason_files.append(group_drop_reasons)
                    cmd += ["--drop-remark-reasons-file", str(group_drop_reasons)]
                if args.timeout_reasons_file:
                    group_timeout_reasons = (
                        workdir / f"pairwise-timeout-reasons-{group_index}.tsv"
                    )
                    timeout_reason_files.append(group_timeout_reasons)
                    cmd += ["--timeout-reasons-file", str(group_timeout_reasons)]
                if args.resolved_contracts_file:
                    group_resolved_contracts = (
                        workdir / f"pairwise-resolved-contracts-{group_index}.tsv"
                    )
                    resolved_contract_files.append(group_resolved_contracts)
                    cmd += ["--resolved-contracts-file", str(group_resolved_contracts)]
                if (
                    args.assertion_results_file
                    or args.fpv_summary_file
                    or args.assertion_granular
                ):
                    group_assertion_results = (
                        workdir / f"pairwise-assertion-results-{group_index}.tsv"
                    )
                    assertion_result_files.append(group_assertion_results)
                    cmd += [
                        "--assertion-results-file",
                        str(group_assertion_results),
                    ]
                if (
                    args.cover_results_file
                    or args.fpv_summary_file
                    or args.cover_granular
                ):
                    group_cover_results = (
                        workdir / f"pairwise-cover-results-{group_index}.tsv"
                    )
                    cover_result_files.append(group_cover_results)
                    cmd += [
                        "--cover-results-file",
                        str(group_cover_results),
                    ]
                if args.assertion_granular:
                    cmd.append("--assertion-granular")
                    if assertion_granular_max > 0:
                        cmd += ["--assertion-granular-max", str(assertion_granular_max)]
                if args.cover_granular:
                    cmd.append("--cover-granular")

                cmd_env = os.environ.copy()
                policy_passes: list[str] = []
                if stopat_selectors:
                    selector_list = ",".join(stopat_selectors)
                    policy_passes.append(
                        f"--hw-stopat-symbolic=targets={selector_list}"
                    )
                if blackbox_modules:
                    module_list = ",".join(blackbox_modules)
                    policy_passes.append(
                        f"--hw-externalize-modules=module-names={module_list}"
                    )
                if policy_passes:
                    cmd_env["BMC_PREPARE_CORE_PASSES"] = " ".join(
                        [*policy_passes, base_prepare_core_passes]
                    ).strip()
                    print(
                        "opentitan FPV BMC: applying task policy via "
                        f"BMC_PREPARE_CORE_PASSES={shlex.quote(cmd_env['BMC_PREPARE_CORE_PASSES'])}",
                        flush=True,
                    )
                pairwise_rc = max(
                    pairwise_rc, subprocess.run(cmd, check=False, env=cmd_env).returncode
                )

        merge_plain_files(drop_case_files, args.drop_remark_cases_file)
        merge_plain_files(drop_reason_files, args.drop_remark_reasons_file)
        merge_plain_files(timeout_reason_files, args.timeout_reasons_file)
        merge_plain_files(assertion_result_files, args.assertion_results_file)
        merge_plain_files(cover_result_files, args.cover_results_file)
        merge_resolved_contract_files(resolved_contract_files, args.resolved_contracts_file)

        merged_rows = list(pre_rows)
        for pairwise_results in pairwise_result_files:
            if not pairwise_results.exists():
                continue
            for line in pairwise_results.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                merged_rows.append(tuple(line.split("\t")))

        if not merged_rows:
            print("No OpenTitan FPV BMC cases selected.", file=sys.stderr)
            return 1

        merged_rows.sort(key=lambda row: (row[1] if len(row) > 1 else "", row[0]))
        if args.results_file:
            out_path = Path(args.results_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as handle:
                for row in merged_rows:
                    handle.write("\t".join(row) + "\n")
            print(f"results: {out_path}", flush=True)

        merged_assertion_rows: list[tuple[str, ...]] = []
        for assertion_path in assertion_result_files:
            if not assertion_path.exists():
                continue
            for line in assertion_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                merged_assertion_rows.append(tuple(line.split("\t")))
        merged_cover_rows: list[tuple[str, ...]] = []
        for cover_path in cover_result_files:
            if not cover_path.exists():
                continue
            for line in cover_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                merged_cover_rows.append(tuple(line.split("\t")))

        if args.fpv_summary_file:
            fpv_summary_path = Path(args.fpv_summary_file)
            write_fpv_summary(
                merged_rows, merged_assertion_rows, merged_cover_rows, fpv_summary_path
            )
            print(f"fpv summary: {fpv_summary_path}", flush=True)
            if args.fpv_summary_baseline_file:
                baseline_path = Path(args.fpv_summary_baseline_file).resolve()
                current_path = fpv_summary_path.resolve()
                baseline = read_fpv_summary(baseline_path)
                current = read_fpv_summary(current_path)

                allow_exact: set[str] = set()
                allow_prefix: list[str] = []
                allow_regex: list[re.Pattern[str]] = []
                if args.fpv_summary_drift_allowlist_file:
                    allow_exact, allow_prefix, allow_regex = load_allowlist(
                        Path(args.fpv_summary_drift_allowlist_file).resolve()
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
                    for kind, before, after in [
                        ("total_assertions", b.total_assertions, c.total_assertions),
                        ("proven", b.proven, c.proven),
                        ("failing", b.failing, c.failing),
                        ("vacuous", b.vacuous, c.vacuous),
                        ("covered", b.covered, c.covered),
                        ("unreachable", b.unreachable, c.unreachable),
                        ("unknown", b.unknown, c.unknown),
                        ("error", b.error, c.error),
                        ("timeout", b.timeout, c.timeout),
                        ("skipped", b.skipped, c.skipped),
                    ]:
                        if before != after:
                            drift_rows.append((target, kind, before, after))

                if args.fpv_summary_drift_file:
                    drift_path = Path(args.fpv_summary_drift_file).resolve()
                    write_fpv_summary_drift(drift_path, drift_rows)
                    print(f"fpv summary drift: {drift_path}", flush=True)

                if drift_rows:
                    sample = ", ".join(
                        f"{target}:{kind}" for target, kind, _, _ in drift_rows[:6]
                    )
                    if len(drift_rows) > 6:
                        sample += ", ..."
                    message = (
                        "opentitan fpv summary drift detected: "
                        f"rows={len(drift_rows)} sample=[{sample}] "
                        f"baseline={baseline_path} current={current_path}"
                    )
                    if args.fail_on_fpv_summary_drift:
                        print(message, file=sys.stderr)
                        return 1
                    print(f"warning: {message}", file=sys.stderr)
                else:
                    print(
                        (
                            "opentitan fpv summary drift check passed: "
                            f"targets={len(current)} baseline={baseline_path} "
                            f"current={current_path}"
                        ),
                        file=sys.stderr,
                    )

        total, passed, failed, xfailed, xpassed, errored, skipped = summarize(
            merged_rows
        )
        print(
            "opentitan FPV BMC summary: "
            f"total={total} pass={passed} fail={failed} xfail={xfailed} "
            f"xpass={xpassed} error={errored} skip={skipped}",
            flush=True,
        )

        if pairwise_rc != 0:
            return pairwise_rc
        if failed or errored or xpassed:
            return 1
        return 0
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

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
    blackbox_modules: tuple[str, ...]
    toplevels: tuple[str, ...]
    files: tuple[str, ...]
    include_dirs: tuple[str, ...]
    defines: tuple[str, ...]


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

    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        keep_workdir = True
    else:
        workdir = Path(tempfile.mkdtemp(prefix="opentitan-fpv-bmc-"))
        keep_workdir = args.keep_workdir

    pre_rows: list[tuple[str, ...]] = []
    grouped_case_lines: dict[tuple[str, ...], list[str]] = {}

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
        if row.stopat_mode == "task_defined" and row.stopat_count > 0:
            add_contract_error(row, "unsupported_stopat_injection")
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
            contract_source = (
                f"fpv_target:{row.target_name};"
                f"task_profile:{row.task_profile or 'unknown'};"
                f"stopat_mode:{row.stopat_mode or 'none'};"
                f"blackbox_policy:{blackbox_policy}"
            )
            grouped_case_lines.setdefault(row.blackbox_modules, []).append(
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
            for group_index, blackbox_modules in enumerate(sorted(grouped_case_lines)):
                group_cases = grouped_case_lines[blackbox_modules]
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

                cmd_env = os.environ.copy()
                if blackbox_modules:
                    module_list = ",".join(blackbox_modules)
                    externalize_pass = (
                        f"--hw-externalize-modules=module-names={module_list}"
                    )
                    cmd_env["BMC_PREPARE_CORE_PASSES"] = (
                        f"{externalize_pass} {base_prepare_core_passes}".strip()
                    )
                    print(
                        "opentitan FPV BMC: applying blackbox policy via "
                        f"BMC_PREPARE_CORE_PASSES={shlex.quote(cmd_env['BMC_PREPARE_CORE_PASSES'])}",
                        flush=True,
                    )
                pairwise_rc = max(
                    pairwise_rc, subprocess.run(cmd, check=False, env=cmd_env).returncode
                )

        merge_plain_files(drop_case_files, args.drop_remark_cases_file)
        merge_plain_files(drop_reason_files, args.drop_remark_reasons_file)
        merge_plain_files(timeout_reason_files, args.timeout_reasons_file)
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

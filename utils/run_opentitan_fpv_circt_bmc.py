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
    setup_status: str
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
        "setup_status",
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
                setup_status=setup_status,
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

    pairwise_results = workdir / "pairwise-results.tsv"
    cases_file = workdir / "pairwise-cases.tsv"
    pre_rows: list[tuple[str, ...]] = []
    case_lines: list[str] = []

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
            contract_source = (
                f"fpv_target:{row.target_name};task_profile:{row.task_profile or 'unknown'}"
            )
            case_lines.append(
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
    if case_lines and not pairwise_runner.is_file():
        fail(f"missing pairwise runner: {pairwise_runner}")

    pairwise_rc = 0
    try:
        if case_lines:
            cases_file.write_text("\n".join(case_lines) + "\n", encoding="utf-8")
            cmd = [
                sys.executable,
                str(pairwise_runner),
                "--cases-file",
                str(cases_file),
                "--suite-name",
                "opentitan",
                "--mode-label",
                mode_label,
                "--bound",
                str(bound),
                "--ignore-asserts-until",
                str(ignore_asserts_until),
                "--workdir",
                str(workdir / "pairwise-work"),
                "--keep-workdir",
                "--results-file",
                str(pairwise_results),
            ]
            if args.drop_remark_cases_file:
                cmd += ["--drop-remark-cases-file", args.drop_remark_cases_file]
            if args.drop_remark_reasons_file:
                cmd += ["--drop-remark-reasons-file", args.drop_remark_reasons_file]
            if args.timeout_reasons_file:
                cmd += ["--timeout-reasons-file", args.timeout_reasons_file]
            if args.resolved_contracts_file:
                cmd += ["--resolved-contracts-file", args.resolved_contracts_file]
            pairwise_rc = subprocess.run(cmd, check=False).returncode

        merged_rows = list(pre_rows)
        if pairwise_results.exists():
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

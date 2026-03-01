#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run baseline manifest commands repeatedly and emit drift reports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

FORMAL_LIB = Path(__file__).resolve().parent / "lib"
if str(FORMAL_LIB) not in sys.path:
    sys.path.insert(0, str(FORMAL_LIB))

from baseline_manifest import (  # noqa: E402
    ManifestCommand,
    ManifestValidationError,
    load_manifest_commands,
)
from runner_common import write_log  # noqa: E402


def fail(msg: str) -> NoReturn:
    raise SystemExit(msg)


def run_manifest_command(
    *,
    entry: ManifestCommand,
    run_index: int,
    command_dir: Path,
    default_command_cwd: Path,
    default_command_timeout_secs: int,
    max_log_bytes: int,
) -> tuple[int, Path, Path, Path]:
    command_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = command_dir / "results.tsv"
    out_jsonl = command_dir / "results.jsonl"
    log_path = command_dir / "command.log"
    env = os.environ.copy()
    env["OUT"] = str(out_tsv)
    env["FORMAL_RESULTS_JSONL_OUT"] = str(out_jsonl)
    env["FORMAL_BASELINE_RUN_INDEX"] = str(run_index)
    env["FORMAL_BASELINE_COMMAND_INDEX"] = str(entry.command_index)
    env["FORMAL_BASELINE_SUITE"] = entry.suite
    env["FORMAL_BASELINE_MODE"] = entry.mode
    env["FORMAL_BASELINE_CASE_LABEL"] = entry.case_label
    command_cwd = Path(entry.cwd).resolve() if entry.cwd else default_command_cwd
    timeout_secs = (
        entry.timeout_secs if entry.timeout_secs > 0 else default_command_timeout_secs
    )
    try:
        proc = subprocess.run(
            entry.command,
            shell=True,
            executable="/bin/bash",
            cwd=str(command_cwd),
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_secs if timeout_secs > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        stderr: str | bytes | None = exc.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        timeout_note = (
            "[capture_formal_baseline] command timeout after "
            f"{timeout_secs}s\n"
        )
        write_log(
            log_path,
            exc.stdout,
            f"{stderr or ''}{timeout_note}",
            max_log_bytes=max_log_bytes,
            truncation_label="capture_formal_baseline",
        )
        out_tsv.write_text("", encoding="utf-8")
        out_jsonl.write_text("", encoding="utf-8")
        return 124, out_tsv, out_jsonl, log_path
    write_log(
        log_path,
        proc.stdout,
        proc.stderr,
        max_log_bytes=max_log_bytes,
        truncation_label="capture_formal_baseline",
    )
    return proc.returncode, out_tsv, out_jsonl, log_path


def compare_drift(
    *,
    drift_script: Path,
    reference_jsonl: Path,
    candidate_jsonl: Path,
    out_tsv: Path,
    summary_json: Path,
    fail_on_status_drift: bool,
    fail_on_missing_case: bool,
    fail_on_new_case: bool,
) -> int:
    cmd = [
        sys.executable,
        str(drift_script),
        "--reference-jsonl",
        str(reference_jsonl),
        "--candidate-jsonl",
        str(candidate_jsonl),
        "--out-tsv",
        str(out_tsv),
        "--summary-json",
        str(summary_json),
    ]
    if fail_on_status_drift:
        cmd.append("--fail-on-status-drift")
    if fail_on_missing_case:
        cmd.append("--fail-on-missing-case")
    if fail_on_new_case:
        cmd.append("--fail-on-new-case")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def validate_results_schema(
    *,
    validator_script: Path,
    jsonl_path: Path,
    summary_json: Path,
    strict_contract: bool,
) -> int:
    cmd = [
        sys.executable,
        str(validator_script),
        "--jsonl",
        str(jsonl_path),
        "--summary-json",
        str(summary_json),
    ]
    if strict_contract:
        cmd.append("--strict-contract")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture repeated formal baseline runs and report drift."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--command-cwd",
        default=".",
        help="Default working directory for manifest commands (default: current directory).",
    )
    parser.add_argument(
        "--command-timeout-secs",
        type=int,
        default=0,
        help=(
            "Default timeout in seconds for each manifest command "
            "(0 disables timeout)."
        ),
    )
    parser.add_argument(
        "--max-log-bytes",
        type=int,
        default=0,
        help=(
            "Maximum bytes per command log file in capture output "
            "(0 disables truncation)."
        ),
    )
    parser.add_argument(
        "--validate-results-schema",
        action="store_true",
        help=(
            "Validate each successful command JSONL output with "
            "validate_formal_results_schema.py."
        ),
    )
    parser.add_argument(
        "--validate-results-schema-strict-contract",
        action="store_true",
        help=(
            "When schema validation is enabled, also enforce strict "
            "cross-row contract checks."
        ),
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--stop-on-command-failure", action="store_true")
    parser.add_argument("--fail-on-status-drift", action="store_true")
    parser.add_argument("--fail-on-missing-case", action="store_true")
    parser.add_argument("--fail-on-new-case", action="store_true")
    parser.add_argument(
        "--dashboard-summary-json",
        default="",
        help=(
            "Optional schema-only dashboard summary JSON path. If set, "
            "capture aggregates successful JSONL command outputs via "
            "build_formal_dashboard_inputs.py."
        ),
    )
    parser.add_argument("--dashboard-status-tsv", default="")
    parser.add_argument("--dashboard-reason-tsv", default="")
    parser.add_argument("--dashboard-top-timeout-cases-tsv", default="")
    parser.add_argument("--dashboard-top-timeout-reasons-tsv", default="")
    parser.add_argument(
        "--dashboard-top-timeout-cases-limit",
        type=int,
        default=20,
        help="Top timeout frontier case rows to keep in dashboard outputs.",
    )
    parser.add_argument(
        "--dashboard-top-timeout-reasons-limit",
        type=int,
        default=20,
        help="Top timeout reason rows to keep in dashboard outputs.",
    )
    parser.add_argument(
        "--dashboard-include-nonsolver-timeouts",
        action="store_true",
        help=(
            "Include non-solver timeout rows when building optional dashboard "
            "summary outputs."
        ),
    )
    args = parser.parse_args()

    if args.repeat < 1:
        fail("--repeat must be >= 1")

    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    default_command_cwd = Path(args.command_cwd).resolve()
    if not default_command_cwd.is_dir():
        fail(f"default command cwd not found: {default_command_cwd}")
    if args.command_timeout_secs < 0:
        fail("--command-timeout-secs must be >= 0")
    if args.max_log_bytes < 0:
        fail("--max-log-bytes must be >= 0")
    if args.validate_results_schema_strict_contract and not args.validate_results_schema:
        fail(
            "--validate-results-schema-strict-contract requires "
            "--validate-results-schema"
        )
    if args.dashboard_top_timeout_cases_limit < 1:
        fail("--dashboard-top-timeout-cases-limit must be >= 1")
    if args.dashboard_top_timeout_reasons_limit < 1:
        fail("--dashboard-top-timeout-reasons-limit must be >= 1")
    drift_script = (Path(__file__).resolve().parent / "compare_formal_results_drift.py")
    if not drift_script.is_file():
        fail(f"drift comparator script not found: {drift_script}")
    validator_script = (
        Path(__file__).resolve().parent / "validate_formal_results_schema.py"
    )
    dashboard_script = (
        Path(__file__).resolve().parent / "build_formal_dashboard_inputs.py"
    )
    dashboard_requested = bool(
        args.dashboard_summary_json
        or args.dashboard_status_tsv
        or args.dashboard_reason_tsv
        or args.dashboard_top_timeout_cases_tsv
        or args.dashboard_top_timeout_reasons_tsv
        or args.dashboard_include_nonsolver_timeouts
    )
    if dashboard_requested:
        if not args.dashboard_summary_json:
            fail(
                "--dashboard-summary-json is required when any dashboard "
                "output option is set"
            )
        if not dashboard_script.is_file():
            fail(f"dashboard builder script not found: {dashboard_script}")
    if args.validate_results_schema and not validator_script.is_file():
        fail(f"schema validator script not found: {validator_script}")

    try:
        manifest_payload, commands = load_manifest_commands(manifest_path)
    except ManifestValidationError as exc:
        fail(str(exc))
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_manifest_path = out_dir / "manifest.json"
    copied_manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    execution_rows: list[tuple[str, ...]] = []
    dashboard_jsonl_inputs: list[Path] = []
    execution_rc = 0
    for run_index in range(1, args.repeat + 1):
        run_dir = out_dir / f"run-{run_index:02d}"
        for entry in commands:
            command_dir = run_dir / entry.case_label
            returncode, out_tsv, out_jsonl, log_path = run_manifest_command(
                entry=entry,
                run_index=run_index,
                command_dir=command_dir,
                default_command_cwd=default_command_cwd,
                default_command_timeout_secs=args.command_timeout_secs,
                max_log_bytes=args.max_log_bytes,
            )
            expected_returncodes = ",".join(
                str(code) for code in entry.expected_returncodes
            )
            command_ok = returncode in entry.expected_returncodes
            schema_validation_rc = 0
            schema_summary_path = ""
            should_validate_schema = False
            if args.validate_results_schema and command_ok:
                if returncode == 0:
                    should_validate_schema = True
                elif out_jsonl.is_file() and out_jsonl.stat().st_size > 0:
                    # For expected non-zero lanes (for example bounded timeout
                    # frontier probes), validate schema when a JSONL payload is
                    # present; keep empty timeout outputs as non-fatal.
                    should_validate_schema = True
            if should_validate_schema:
                schema_summary = command_dir / "results_schema_summary.json"
                schema_validation_rc = validate_results_schema(
                    validator_script=validator_script,
                    jsonl_path=out_jsonl,
                    summary_json=schema_summary,
                    strict_contract=args.validate_results_schema_strict_contract,
                )
                schema_summary_path = str(schema_summary)
            execution_rows.append(
                (
                    str(run_index),
                    str(entry.command_index),
                    entry.suite,
                    entry.mode,
                    entry.case_label,
                    str(returncode),
                    entry.cwd or str(default_command_cwd),
                    str(
                        entry.timeout_secs
                        if entry.timeout_secs > 0
                        else args.command_timeout_secs
                    ),
                    expected_returncodes,
                    entry.command,
                    str(out_tsv),
                    str(out_jsonl),
                    str(log_path),
                    str(schema_validation_rc),
                    schema_summary_path,
                )
            )
            if command_ok and out_jsonl.is_file():
                dashboard_jsonl_inputs.append(out_jsonl.resolve())
            if not command_ok or schema_validation_rc != 0:
                execution_rc = 1
                if args.stop_on_command_failure:
                    break
        if execution_rc != 0 and args.stop_on_command_failure:
            break

    execution_tsv = out_dir / "execution.tsv"
    with execution_tsv.open("w", encoding="utf-8") as handle:
        handle.write(
            "run_index\tcommand_index\tsuite\tmode\tcase_label\treturncode\t"
            "command_cwd\tcommand_timeout_secs\texpected_returncodes\tcommand\tresults_tsv\t"
            "results_jsonl\tlog_path\tschema_validation_rc\t"
            "schema_summary_json\n"
        )
        for row in execution_rows:
            handle.write("\t".join(row))
            handle.write("\n")

    if dashboard_requested:
        if not dashboard_jsonl_inputs:
            fail(
                "dashboard outputs requested but no expected-returncode "
                "command JSONL outputs were captured"
            )
        dashboard_cmd = [sys.executable, str(dashboard_script)]
        for jsonl_path in dashboard_jsonl_inputs:
            dashboard_cmd.extend(["--jsonl", str(jsonl_path)])
        dashboard_cmd.extend(
            [
                "--summary-json",
                str(Path(args.dashboard_summary_json).resolve()),
                "--top-timeout-cases-limit",
                str(args.dashboard_top_timeout_cases_limit),
                "--top-timeout-reasons-limit",
                str(args.dashboard_top_timeout_reasons_limit),
            ]
        )
        if args.dashboard_status_tsv:
            dashboard_cmd.extend(
                ["--status-tsv", str(Path(args.dashboard_status_tsv).resolve())]
            )
        if args.dashboard_reason_tsv:
            dashboard_cmd.extend(
                ["--reason-tsv", str(Path(args.dashboard_reason_tsv).resolve())]
            )
        if args.dashboard_top_timeout_cases_tsv:
            dashboard_cmd.extend(
                [
                    "--top-timeout-cases-tsv",
                    str(Path(args.dashboard_top_timeout_cases_tsv).resolve()),
                ]
            )
        if args.dashboard_top_timeout_reasons_tsv:
            dashboard_cmd.extend(
                [
                    "--top-timeout-reasons-tsv",
                    str(Path(args.dashboard_top_timeout_reasons_tsv).resolve()),
                ]
            )
        if args.dashboard_include_nonsolver_timeouts:
            dashboard_cmd.append("--include-nonsolver-timeouts")
        dashboard_proc = subprocess.run(dashboard_cmd, check=False)
        if dashboard_proc.returncode != 0:
            execution_rc = 1

    drift_rows: list[tuple[str, ...]] = []
    drift_rc = 0
    if args.repeat > 1:
        drift_root = out_dir / "drift"
        drift_root.mkdir(parents=True, exist_ok=True)
        for run_index in range(2, args.repeat + 1):
            for entry in commands:
                ref_jsonl = out_dir / "run-01" / entry.case_label / "results.jsonl"
                cand_jsonl = (
                    out_dir
                    / f"run-{run_index:02d}"
                    / entry.case_label
                    / "results.jsonl"
                )
                drift_tsv = (
                    drift_root
                    / f"{entry.case_label}.run01-vs-run{run_index:02d}.tsv"
                )
                drift_summary = (
                    drift_root
                    / f"{entry.case_label}.run01-vs-run{run_index:02d}.summary.json"
                )
                rc = compare_drift(
                    drift_script=drift_script,
                    reference_jsonl=ref_jsonl,
                    candidate_jsonl=cand_jsonl,
                    out_tsv=drift_tsv,
                    summary_json=drift_summary,
                    fail_on_status_drift=args.fail_on_status_drift,
                    fail_on_missing_case=args.fail_on_missing_case,
                    fail_on_new_case=args.fail_on_new_case,
                )
                drift_rows.append(
                    (
                        "1",
                        str(run_index),
                        str(entry.command_index),
                        entry.suite,
                        entry.mode,
                        entry.case_label,
                        str(rc),
                        str(drift_tsv),
                        str(drift_summary),
                    )
                )
                if rc != 0:
                    drift_rc = 1

    drift_tsv_path = out_dir / "drift.tsv"
    with drift_tsv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "reference_run\tcandidate_run\tcommand_index\tsuite\tmode\t"
            "case_label\treturncode\tdrift_tsv\tdrift_summary_json\n"
        )
        for row in drift_rows:
            handle.write("\t".join(row))
            handle.write("\n")

    return max(execution_rc, drift_rc)


if __name__ == "__main__":
    raise SystemExit(main())

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


def fail(msg: str) -> NoReturn:
    raise SystemExit(msg)


def coerce_text_output(raw: str | bytes | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


def write_log_text(log_path: Path, log_data: str, max_log_bytes: int) -> None:
    if max_log_bytes <= 0:
        log_path.write_text(log_data, encoding="utf-8")
        return
    encoded = log_data.encode("utf-8", errors="replace")
    if len(encoded) <= max_log_bytes:
        log_path.write_bytes(encoded)
        return
    notice = (
        "\n[capture_formal_baseline] log truncated from "
        f"{len(encoded)} to {max_log_bytes} bytes\n"
    ).encode("utf-8")
    if max_log_bytes <= len(notice):
        log_path.write_bytes(encoded[:max_log_bytes])
        return
    keep = max_log_bytes - len(notice)
    log_path.write_bytes(encoded[:keep] + notice)


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
        stdout = coerce_text_output(exc.stdout)
        stderr = coerce_text_output(exc.stderr)
        log_data = ""
        if stdout:
            log_data += stdout
            if not log_data.endswith("\n"):
                log_data += "\n"
        if stderr:
            log_data += stderr
            if not log_data.endswith("\n"):
                log_data += "\n"
        log_data += (
            "[capture_formal_baseline] command timeout after "
            f"{timeout_secs}s\n"
        )
        write_log_text(log_path, log_data, max_log_bytes)
        out_tsv.write_text("", encoding="utf-8")
        out_jsonl.write_text("", encoding="utf-8")
        return 124, out_tsv, out_jsonl, log_path
    log_data = ""
    if proc.stdout:
        log_data += proc.stdout
        if not log_data.endswith("\n"):
            log_data += "\n"
    if proc.stderr:
        log_data += proc.stderr
    write_log_text(log_path, log_data, max_log_bytes)
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
    *, validator_script: Path, jsonl_path: Path, summary_json: Path
) -> int:
    cmd = [
        sys.executable,
        str(validator_script),
        "--jsonl",
        str(jsonl_path),
        "--summary-json",
        str(summary_json),
    ]
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
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--stop-on-command-failure", action="store_true")
    parser.add_argument("--fail-on-status-drift", action="store_true")
    parser.add_argument("--fail-on-missing-case", action="store_true")
    parser.add_argument("--fail-on-new-case", action="store_true")
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
    drift_script = (Path(__file__).resolve().parent / "compare_formal_results_drift.py")
    if not drift_script.is_file():
        fail(f"drift comparator script not found: {drift_script}")
    validator_script = (
        Path(__file__).resolve().parent / "validate_formal_results_schema.py"
    )
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
            schema_validation_rc = 0
            schema_summary_path = ""
            if args.validate_results_schema and returncode == 0:
                schema_summary = command_dir / "results_schema_summary.json"
                schema_validation_rc = validate_results_schema(
                    validator_script=validator_script,
                    jsonl_path=out_jsonl,
                    summary_json=schema_summary,
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
                    entry.command,
                    str(out_tsv),
                    str(out_jsonl),
                    str(log_path),
                    str(schema_validation_rc),
                    schema_summary_path,
                )
            )
            if returncode != 0 or schema_validation_rc != 0:
                execution_rc = 1
                if args.stop_on_command_failure:
                    break
        if execution_rc != 0 and args.stop_on_command_failure:
            break

    execution_tsv = out_dir / "execution.tsv"
    with execution_tsv.open("w", encoding="utf-8") as handle:
        handle.write(
            "run_index\tcommand_index\tsuite\tmode\tcase_label\treturncode\t"
            "command_cwd\tcommand_timeout_secs\tcommand\tresults_tsv\t"
            "results_jsonl\tlog_path\tschema_validation_rc\t"
            "schema_summary_json\n"
        )
        for row in execution_rows:
            handle.write("\t".join(row))
            handle.write("\n")

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

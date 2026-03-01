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
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn


def fail(msg: str) -> NoReturn:
    raise SystemExit(msg)


def slugify(token: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9._-]+", "_", token.strip())
    compact = compact.strip("._-")
    return compact or "command"


@dataclass(frozen=True)
class ManifestCommand:
    command_index: int
    suite: str
    mode: str
    case_label: str
    command: str
    cwd: str


def load_manifest_commands(path: Path) -> tuple[dict[str, Any], list[ManifestCommand]]:
    if not path.is_file():
        fail(f"manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid manifest JSON ({path}): {exc}")
    if not isinstance(payload, dict):
        fail(f"invalid manifest root (expected object): {path}")
    raw_commands = payload.get("commands", [])
    if not isinstance(raw_commands, list) or not raw_commands:
        fail("manifest missing non-empty commands list")
    commands: list[ManifestCommand] = []
    for index, raw in enumerate(raw_commands, start=1):
        if not isinstance(raw, dict):
            fail(f"manifest commands[{index}] must be an object")
        suite = str(raw.get("suite", "")).strip()
        mode = str(raw.get("mode", "")).strip()
        command = str(raw.get("command", "")).strip()
        if not suite or not mode or not command:
            fail(
                "manifest command missing required keys "
                f"(suite/mode/command) at index {index}"
            )
        label_seed = (
            str(raw.get("id", "")).strip()
            or str(raw.get("label", "")).strip()
            or f"{suite}_{mode}_{index}"
        )
        commands.append(
            ManifestCommand(
                command_index=index,
                suite=suite,
                mode=mode,
                case_label=slugify(label_seed),
                command=command,
                cwd=str(raw.get("cwd", "")).strip(),
            )
        )
    return payload, commands


def run_manifest_command(
    *,
    entry: ManifestCommand,
    run_index: int,
    command_dir: Path,
    default_command_cwd: Path,
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
    proc = subprocess.run(
        entry.command,
        shell=True,
        executable="/bin/bash",
        cwd=str(command_cwd),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    log_data = ""
    if proc.stdout:
        log_data += proc.stdout
        if not log_data.endswith("\n"):
            log_data += "\n"
    if proc.stderr:
        log_data += proc.stderr
    log_path.write_text(log_data, encoding="utf-8")
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
    drift_script = (Path(__file__).resolve().parent / "compare_formal_results_drift.py")
    if not drift_script.is_file():
        fail(f"drift comparator script not found: {drift_script}")

    manifest_payload, commands = load_manifest_commands(manifest_path)
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
            )
            execution_rows.append(
                (
                    str(run_index),
                    str(entry.command_index),
                    entry.suite,
                    entry.mode,
                    entry.case_label,
                    str(returncode),
                    entry.cwd or str(default_command_cwd),
                    entry.command,
                    str(out_tsv),
                    str(out_jsonl),
                    str(log_path),
                )
            )
            if returncode != 0:
                execution_rc = 1
                if args.stop_on_command_failure:
                    break
        if execution_rc != 0 and args.stop_on_command_failure:
            break

    execution_tsv = out_dir / "execution.tsv"
    with execution_tsv.open("w", encoding="utf-8") as handle:
        handle.write(
            "run_index\tcommand_index\tsuite\tmode\tcase_label\treturncode\t"
            "command_cwd\tcommand\tresults_tsv\tresults_jsonl\tlog_path\n"
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

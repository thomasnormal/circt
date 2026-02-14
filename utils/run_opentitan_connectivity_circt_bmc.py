#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run OpenTitan connectivity rules through the generic pairwise BMC runner.

This utility consumes connectivity manifests emitted by
`select_opentitan_connectivity_cfg.py`, synthesizes per-connection bind-check
modules, and delegates execution to `run_pairwise_circt_bmc.py`.
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
from typing import Any

import yaml


@dataclass(frozen=True)
class ConnectivityTarget:
    target_name: str
    fusesoc_core: str


@dataclass(frozen=True)
class ConnectivityRule:
    rule_id: str
    rule_type: str
    csv_file: str
    csv_row: int
    rule_name: str
    src_block: str
    src_signal: str
    dest_block: str
    dest_signal: str


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_nonnegative_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        fail(f"invalid {name}: {raw}")
    if value < 0:
        fail(f"invalid {name}: {raw}")
    return value


def parse_target_manifest(path: Path) -> ConnectivityTarget:
    if not path.is_file():
        fail(f"connectivity target manifest not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"connectivity target manifest missing header row: {path}")
        required = {"target_name", "fusesoc_core"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            fail(
                f"connectivity target manifest missing columns {missing}: {path} "
                f"(found: {reader.fieldnames})"
            )
        rows = list(reader)
    if not rows:
        fail(f"connectivity target manifest has no rows: {path}")
    if len(rows) != 1:
        fail(
            f"connectivity target manifest expected exactly 1 target row: {path} "
            f"(found {len(rows)})"
        )
    row = rows[0]
    target_name = (row.get("target_name") or "").strip()
    fusesoc_core = (row.get("fusesoc_core") or "").strip()
    if not target_name:
        fail(f"connectivity target row missing target_name in {path}")
    if not fusesoc_core:
        fail(f"connectivity target row missing fusesoc_core in {path}")
    return ConnectivityTarget(target_name=target_name, fusesoc_core=fusesoc_core)


def parse_rules_manifest(path: Path) -> list[ConnectivityRule]:
    if not path.is_file():
        fail(f"connectivity rules manifest not found: {path}")
    out: list[ConnectivityRule] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"connectivity rules manifest missing header row: {path}")
        required = {
            "rule_id",
            "rule_type",
            "csv_file",
            "csv_row",
            "rule_name",
            "src_block",
            "src_signal",
            "dest_block",
            "dest_signal",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            fail(
                f"connectivity rules manifest missing columns {missing}: {path} "
                f"(found: {reader.fieldnames})"
            )
        for idx, row in enumerate(reader, start=2):
            rule_id = (row.get("rule_id") or "").strip()
            if not rule_id:
                continue
            rule_type = (row.get("rule_type") or "").strip().upper()
            csv_file = (row.get("csv_file") or "").strip()
            rule_name = (row.get("rule_name") or "").strip()
            csv_row = parse_nonnegative_int((row.get("csv_row") or "0").strip(), "csv_row")
            out.append(
                ConnectivityRule(
                    rule_id=rule_id,
                    rule_type=rule_type,
                    csv_file=csv_file,
                    csv_row=csv_row,
                    rule_name=rule_name,
                    src_block=(row.get("src_block") or "").strip(),
                    src_signal=(row.get("src_signal") or "").strip(),
                    dest_block=(row.get("dest_block") or "").strip(),
                    dest_signal=(row.get("dest_signal") or "").strip(),
                )
            )
    return out


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if isinstance(value, list):
        out: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                token = entry.strip()
                if token:
                    out.append(token)
            else:
                out.append(str(entry))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for key in sorted(value.keys()):
            raw = value[key]
            if raw is None or raw == "":
                out.append(str(key))
            else:
                out.append(f"{key}={raw}")
        return out
    return [str(value)]


def run_fusesoc_setup(
    fusesoc_bin: str,
    opentitan_root: Path,
    fusesoc_core: str,
    job_dir: Path,
    target: str,
    tool: str,
) -> tuple[int, Path]:
    log_path = job_dir / "fusesoc-setup.log"
    cmd = [
        fusesoc_bin,
        "--cores-root",
        str(opentitan_root),
        "run",
        "--target",
        target,
        "--tool",
        tool,
        "--setup",
        fusesoc_core,
    ]
    proc = subprocess.run(
        cmd,
        cwd=job_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    payload = stdout
    if payload and not payload.endswith("\n"):
        payload += "\n"
    payload += stderr
    log_path.write_text(payload, encoding="utf-8")
    return proc.returncode, log_path


def locate_eda_yml(job_dir: Path) -> Path | None:
    candidates = sorted(
        job_dir.glob("build/**/*.eda.yml"),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0]


def parse_toplevels(value: Any) -> list[str]:
    parts = normalize_string_list(value)
    if len(parts) == 1 and "," in parts[0]:
        return [p.strip() for p in parts[0].split(",") if p.strip()]
    return [p for p in parts if p]


def resolve_eda_paths(entries: Any, eda_dir: Path) -> tuple[list[str], list[str]]:
    if not isinstance(entries, list):
        return [], []
    files: list[str] = []
    include_dirs: list[str] = []
    seen_incdirs: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        file_path = (eda_dir / name).resolve(strict=False)
        file_text = str(file_path)
        files.append(file_text)
        if entry.get("is_include_file"):
            incdir = str(file_path.parent)
            if incdir not in seen_incdirs:
                include_dirs.append(incdir)
                seen_incdirs.add(incdir)
    return files, include_dirs


def sanitize_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", token)


def block_signal_expr(block: str, signal: str, top_module: str) -> str:
    b = block.strip()
    s = signal.strip()
    if b:
        if b == top_module:
            b = ""
        elif b.startswith(top_module + "."):
            b = b[len(top_module) + 1 :]
    if b and s:
        return f"{b}.{s}"
    return b or s


def synthesize_rule_checker(
    out_path: Path,
    module_name: str,
    bind_top: str,
    src_expr: str,
    dst_expr: str,
    rule_id: str,
) -> None:
    body = f"""// Auto-generated connectivity check for {rule_id}
module {module_name};
  always_comb begin
    assert (({src_expr}) === ({dst_expr}));
  end
endmodule

bind {bind_top} {module_name} {module_name}_inst();
"""
    out_path.write_text(body, encoding="utf-8")


def build_case_manifest(
    path: Path,
    top_module: str,
    base_source_files: list[str],
    include_dirs: list[str],
    defines: list[str],
    rules: list[ConnectivityRule],
    generated_sv_files: dict[str, Path],
    bound: int,
    ignore_asserts_until: int,
) -> None:
    lines: list[str] = []
    for rule in rules:
        generated = generated_sv_files.get(rule.rule_id)
        if generated is None:
            continue
        case_id = f"connectivity::{rule.rule_id}"
        case_path = f"{rule.csv_file}:{rule.csv_row}"
        source_files = base_source_files + [str(generated)]
        row = [
            case_id,
            top_module,
            ";".join(source_files),
            ";".join(include_dirs),
            case_path,
            "",
            "default",
            str(bound),
            str(ignore_asserts_until),
            "default",
            "default",
            "",
            "connectivity_rule",
            ";".join(defines),
        ]
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OpenTitan connectivity rules through pairwise CIRCT BMC."
    )
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--rules-manifest", required=True)
    parser.add_argument("--opentitan-root", required=True)
    parser.add_argument("--workdir", default="")
    parser.add_argument(
        "--rule-filter",
        default="",
        help="Optional regex filter applied to rule_id and rule_name.",
    )
    parser.add_argument(
        "--rule-shard-count",
        default=os.environ.get("BMC_RULE_SHARD_COUNT", "1"),
        help="Deterministic rule shard count (default: 1).",
    )
    parser.add_argument(
        "--rule-shard-index",
        default=os.environ.get("BMC_RULE_SHARD_INDEX", "0"),
        help="Deterministic rule shard index in [0, count).",
    )
    parser.add_argument(
        "--fusesoc-bin",
        default="fusesoc",
        help="FuseSoC executable (default: fusesoc).",
    )
    parser.add_argument(
        "--fusesoc-target",
        default="formal",
        help="FuseSoC target (default: formal).",
    )
    parser.add_argument(
        "--fusesoc-tool",
        default="symbiyosys",
        help="FuseSoC tool (default: symbiyosys).",
    )
    parser.add_argument(
        "--pairwise-runner",
        default="",
        help="Optional path to run_pairwise_circt_bmc.py.",
    )
    parser.add_argument(
        "--mode-label",
        default=os.environ.get("BMC_MODE_LABEL", "CONNECTIVITY_BMC"),
        help="Mode label passed to pairwise runner results.",
    )
    parser.add_argument(
        "--suite-name",
        default="opentitan-connectivity",
        help="Suite name passed to pairwise runner results.",
    )
    parser.add_argument(
        "--bound",
        default=os.environ.get("BOUND", "1"),
        help="Bound passed to generated per-rule cases.",
    )
    parser.add_argument(
        "--ignore-asserts-until",
        default=os.environ.get("IGNORE_ASSERTS_UNTIL", "0"),
        help="ignore_asserts_until passed to generated per-rule cases.",
    )
    parser.add_argument(
        "--results-file",
        default=os.environ.get("OUT", ""),
        help="Output results TSV path (default: env OUT).",
    )
    args = parser.parse_args()

    if not args.results_file:
        fail("missing --results-file (or OUT environment)")

    target_manifest = Path(args.target_manifest).resolve()
    rules_manifest = Path(args.rules_manifest).resolve()
    opentitan_root = Path(args.opentitan_root).resolve()
    results_file = Path(args.results_file).resolve()
    if not opentitan_root.is_dir():
        fail(f"opentitan root not found: {opentitan_root}")

    rule_shard_count = parse_nonnegative_int(args.rule_shard_count, "rule-shard-count")
    rule_shard_index = parse_nonnegative_int(args.rule_shard_index, "rule-shard-index")
    if rule_shard_count < 1:
        fail("invalid --rule-shard-count: expected integer >= 1")
    if rule_shard_index >= rule_shard_count:
        fail("invalid --rule-shard-index: expected value < --rule-shard-count")
    bound = parse_nonnegative_int(args.bound, "bound")
    ignore_asserts_until = parse_nonnegative_int(
        args.ignore_asserts_until, "ignore-asserts-until"
    )

    rule_filter_re: re.Pattern[str] | None = None
    if args.rule_filter:
        try:
            rule_filter_re = re.compile(args.rule_filter)
        except re.error as exc:
            fail(f"invalid --rule-filter: {args.rule_filter}: {exc}")

    target = parse_target_manifest(target_manifest)
    rules = parse_rules_manifest(rules_manifest)
    connection_rules = [r for r in rules if r.rule_type == "CONNECTION"]
    if rule_filter_re is not None:
        connection_rules = [
            r
            for r in connection_rules
            if rule_filter_re.search(r.rule_id) or rule_filter_re.search(r.rule_name)
        ]
    selected_rules = [
        r for idx, r in enumerate(connection_rules) if (idx % rule_shard_count) == rule_shard_index
    ]

    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text("", encoding="utf-8")
    if not selected_rules:
        print(
            "No OpenTitan connectivity BMC cases selected.",
            file=sys.stderr,
            flush=True,
        )
        return 0

    pairwise_runner = (
        Path(args.pairwise_runner).resolve()
        if args.pairwise_runner
        else (Path(__file__).resolve().parent / "run_pairwise_circt_bmc.py")
    )
    if not pairwise_runner.is_file():
        fail(f"pairwise runner not found: {pairwise_runner}")

    if shutil.which(args.fusesoc_bin) is None and not Path(args.fusesoc_bin).exists():
        fail(f"fusesoc executable not found: {args.fusesoc_bin}")

    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="opentitan-conn-bmc-")
        workdir = Path(temp_dir_obj.name)

    try:
        fusesoc_dir = workdir / "fusesoc"
        fusesoc_dir.mkdir(parents=True, exist_ok=True)
        rc, setup_log = run_fusesoc_setup(
            args.fusesoc_bin,
            opentitan_root,
            target.fusesoc_core,
            fusesoc_dir,
            args.fusesoc_target,
            args.fusesoc_tool,
        )
        eda_yml = locate_eda_yml(fusesoc_dir)
        if eda_yml is None:
            fail(
                "failed to locate FuseSoC EDA description for connectivity target "
                f"{target.target_name}; setup_log={setup_log}"
            )
        if rc != 0:
            print(
                "warning: fusesoc setup returned non-zero for connectivity target "
                f"{target.target_name}; continuing with found EDA file {eda_yml}",
                file=sys.stderr,
            )

        eda_obj = yaml.safe_load(eda_yml.read_text(encoding="utf-8")) or {}
        if not isinstance(eda_obj, dict):
            fail(f"invalid eda yml object (not dict): {eda_yml}")
        toplevels = parse_toplevels(eda_obj.get("toplevel"))
        if not toplevels:
            fail(f"no toplevel in EDA description: {eda_yml}")
        top_module = toplevels[0]
        source_files, include_dirs = resolve_eda_paths(eda_obj.get("files"), eda_yml.parent)
        if not source_files:
            fail(f"no source files resolved from EDA description: {eda_yml}")
        explicit_incdirs = [
            str((eda_yml.parent / item).resolve(strict=False))
            for item in normalize_string_list(eda_obj.get("incdirs"))
        ]
        for incdir in explicit_incdirs:
            if incdir not in include_dirs:
                include_dirs.append(incdir)
        defines = normalize_string_list(eda_obj.get("vlogdefine"))

        checks_dir = workdir / "checks"
        checks_dir.mkdir(parents=True, exist_ok=True)
        generated_sv_files: dict[str, Path] = {}
        skipped_rules = 0
        for index, rule in enumerate(selected_rules):
            src_expr = block_signal_expr(rule.src_block, rule.src_signal, top_module)
            dst_expr = block_signal_expr(rule.dest_block, rule.dest_signal, top_module)
            if not src_expr or not dst_expr:
                skipped_rules += 1
                continue
            module_token = sanitize_token(f"conn_rule_{index}_{rule.rule_name}")
            if not module_token:
                module_token = f"conn_rule_{index}"
            module_name = f"__circt_{module_token}"
            out_sv = checks_dir / f"{module_name}.sv"
            synthesize_rule_checker(
                out_sv,
                module_name,
                top_module,
                src_expr,
                dst_expr,
                rule.rule_id,
            )
            generated_sv_files[rule.rule_id] = out_sv

        cases_file = workdir / "connectivity-cases.tsv"
        build_case_manifest(
            cases_file,
            top_module,
            source_files,
            include_dirs,
            defines,
            selected_rules,
            generated_sv_files,
            bound if bound > 0 else 1,
            ignore_asserts_until,
        )
        if not cases_file.read_text(encoding="utf-8").strip():
            print(
                "No OpenTitan connectivity BMC cases selected.",
                file=sys.stderr,
                flush=True,
            )
            return 0

        cmd = [
            sys.executable,
            str(pairwise_runner),
            "--cases-file",
            str(cases_file),
            "--suite-name",
            args.suite_name,
            "--mode-label",
            args.mode_label,
            "--results-file",
            str(results_file),
            "--workdir",
            str(workdir / "pairwise"),
        ]
        resolved_contracts_out = os.environ.get("BMC_RESOLVED_CONTRACTS_OUT", "").strip()
        if resolved_contracts_out:
            cmd.extend(["--resolved-contracts-file", resolved_contracts_out])
        assertion_results_out = os.environ.get("BMC_ASSERTION_RESULTS_OUT", "").strip()
        if assertion_results_out:
            cmd.extend(["--assertion-results-file", assertion_results_out])
        cover_results_out = os.environ.get("BMC_COVER_RESULTS_OUT", "").strip()
        if cover_results_out:
            cmd.extend(["--cover-results-file", cover_results_out])
        drop_cases_out = os.environ.get("BMC_DROP_REMARK_CASES_OUT", "").strip()
        if drop_cases_out:
            cmd.extend(["--drop-remark-cases-file", drop_cases_out])
        drop_reasons_out = os.environ.get("BMC_DROP_REMARK_REASONS_OUT", "").strip()
        if drop_reasons_out:
            cmd.extend(["--drop-remark-reasons-file", drop_reasons_out])
        timeout_reasons_out = os.environ.get("BMC_TIMEOUT_REASON_CASES_OUT", "").strip()
        if timeout_reasons_out:
            cmd.extend(["--timeout-reasons-file", timeout_reasons_out])

        print(
            "opentitan connectivity bmc: "
            f"target={target.target_name} selected_rules={len(selected_rules)} "
            f"generated_cases={len(generated_sv_files)} skipped_rules={skipped_rules} "
            f"top={top_module} shard={rule_shard_index}/{rule_shard_count}",
            file=sys.stderr,
            flush=True,
        )
        proc = subprocess.run(cmd, check=False)
        return proc.returncode
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

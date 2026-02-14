#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run OpenTitan connectivity rules through CIRCT LEC.

This utility consumes connectivity manifests emitted by
`select_opentitan_connectivity_cfg.py`, synthesizes per-connection wrapper
modules, and checks each rule with `circt-lec` (`ref` wrapper vs `impl`
wrapper).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


@dataclass(frozen=True)
class ConnectivityConnectionGroup:
    connection: ConnectivityRule
    conditions: tuple[ConnectivityRule, ...]


@dataclass(frozen=True)
class ConnectivityLECCase:
    case_id: str
    case_path: str
    rule_id: str
    checker_sv: Path
    ref_module: str
    impl_module: str


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
        for row in reader:
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


def build_connection_groups(
    rules: list[ConnectivityRule],
) -> list[ConnectivityConnectionGroup]:
    groups: list[ConnectivityConnectionGroup] = []
    current_connection: ConnectivityRule | None = None
    current_conditions: list[ConnectivityRule] = []
    orphan_conditions: list[ConnectivityRule] = []

    for rule in rules:
        if rule.rule_type == "CONNECTION":
            if current_connection is not None:
                groups.append(
                    ConnectivityConnectionGroup(
                        connection=current_connection,
                        conditions=tuple(current_conditions),
                    )
                )
            current_connection = rule
            current_conditions = []
            continue
        if rule.rule_type != "CONDITION":
            fail(
                "unsupported connectivity rule type in manifest: "
                f"{rule.rule_type} ({rule.rule_id})"
            )
        if current_connection is None or rule.csv_file != current_connection.csv_file:
            orphan_conditions.append(rule)
            continue
        current_conditions.append(rule)

    if current_connection is not None:
        groups.append(
            ConnectivityConnectionGroup(
                connection=current_connection,
                conditions=tuple(current_conditions),
            )
        )

    if orphan_conditions:
        first = orphan_conditions[0]
        fail(
            "orphan CONDITION row without preceding CONNECTION in manifest: "
            f"{first.csv_file}:{first.csv_row} ({first.rule_id})"
        )
    return groups


def group_matches_rule_filter(
    group: ConnectivityConnectionGroup, rule_filter_re: re.Pattern[str] | None
) -> bool:
    if rule_filter_re is None:
        return True
    tokens = [group.connection.rule_id, group.connection.rule_name]
    for condition in group.conditions:
        tokens.append(condition.rule_id)
        tokens.append(condition.rule_name)
    for token in tokens:
        if token and rule_filter_re.search(token):
            return True
    return False


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
    job_dir.mkdir(parents=True, exist_ok=True)
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


def scope_under_instance(expr: str, instance_name: str) -> str:
    token = expr.strip()
    if not token:
        return ""
    if token.startswith(instance_name + "."):
        return token
    return f"{instance_name}.{token}"


def build_condition_guard_expr(
    conditions: tuple[ConnectivityRule, ...],
    top_module: str,
    instance_name: str,
) -> str:
    if not conditions:
        return ""
    true_terms: list[str] = []
    false_terms: list[str] = []
    for condition in conditions:
        cond_expr_raw = block_signal_expr(
            condition.src_block, condition.src_signal, top_module
        )
        if not cond_expr_raw:
            fail(
                "invalid CONDITION row missing signal expression: "
                f"{condition.csv_file}:{condition.csv_row} ({condition.rule_id})"
            )
        cond_expr = scope_under_instance(cond_expr_raw, instance_name)
        expected_true = condition.dest_block.strip()
        if not expected_true:
            fail(
                "invalid CONDITION row missing expected-true value: "
                f"{condition.csv_file}:{condition.csv_row} ({condition.rule_id})"
            )
        true_terms.append(f"(({cond_expr}) === ({expected_true}))")
        expected_false = condition.dest_signal.strip()
        if expected_false:
            false_terms.append(f"(({cond_expr}) === ({expected_false}))")

    true_expr = " && ".join(true_terms)
    if false_terms:
        false_expr = " || ".join(false_terms)
        return f"(({true_expr}) || ({false_expr}))"
    return f"({true_expr})"


def synthesize_rule_wrappers(
    out_path: Path,
    ref_module: str,
    impl_module: str,
    bind_top: str,
    src_expr: str,
    dst_expr: str,
    rule_id: str,
    guard_expr: str,
) -> None:
    assertion_expr = f"(({src_expr}) === ({dst_expr}))"
    if guard_expr:
        assertion_expr = f"((!({guard_expr})) || {assertion_expr})"
    body = f"""// Auto-generated connectivity LEC wrappers for {rule_id}
module {ref_module}(output logic result);
  {bind_top} dut();
  assign result = 1'b1;
endmodule

module {impl_module}(output logic result);
  {bind_top} dut();
  assign result = {assertion_expr};
endmodule
"""
    out_path.write_text(body, encoding="utf-8")


def write_log(path: Path, stdout: str, stderr: str) -> None:
    data = ""
    if stdout:
        data += stdout
        if not data.endswith("\n"):
            data += "\n"
    if stderr:
        data += stderr
    path.write_text(data, encoding="utf-8")


def parse_lec_result(text: str) -> str | None:
    match = re.search(r"LEC_RESULT=(EQ|NEQ|UNKNOWN)", text)
    if match:
        return match.group(1)
    if re.search(r"\bc1 == c2\b", text):
        return "EQ"
    if re.search(r"\bc1 != c2\b", text):
        return "NEQ"
    return None


def parse_lec_diag(text: str) -> str | None:
    match = re.search(r"LEC_DIAG=([A-Z0-9_]+)", text)
    if not match:
        return None
    return match.group(1)


def normalize_drop_reason(line: str) -> str:
    line = line.replace("\t", " ").strip()
    line = re.sub(r"^[^:\n]+:\d+(?::\d+)?:\s*", "", line)
    line = re.sub(r"^[Ww]arning:\s*", "", line)
    line = re.sub(r"\s+", " ", line)
    line = re.sub(r"\d+", "<n>", line)
    line = line.replace(";", ",")
    if len(line) > 240:
        line = line[:240]
    return line


def extract_drop_reasons(log_text: str, pattern: str) -> list[str]:
    reasons: set[str] = set()
    for line in log_text.splitlines():
        if pattern not in line:
            continue
        reason = normalize_drop_reason(line)
        if reason:
            reasons.add(reason)
    return sorted(reasons)


def run_and_log(
    cmd: list[str],
    log_path: Path,
    timeout_secs: int,
    out_path: Path | None = None,
) -> str:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_secs if timeout_secs > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        write_log(log_path, stdout, stderr)
        if out_path is not None:
            out_path.write_text(stdout, encoding="utf-8")
        raise
    write_log(log_path, result.stdout or "", result.stderr or "")
    if out_path is not None:
        out_path.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    return (result.stdout or "") + "\n" + (result.stderr or "")


def compute_contract_fingerprint(fields: list[str]) -> str:
    payload = "\x1f".join(fields).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OpenTitan connectivity rules through CIRCT LEC."
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
        default=os.environ.get("LEC_RULE_SHARD_COUNT", "1"),
        help="Deterministic rule shard count (default: 1).",
    )
    parser.add_argument(
        "--rule-shard-index",
        default=os.environ.get("LEC_RULE_SHARD_INDEX", "0"),
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
        "--mode-label",
        default=os.environ.get("LEC_MODE_LABEL", "CONNECTIVITY_LEC"),
        help="Mode label written to result rows (default: CONNECTIVITY_LEC).",
    )
    parser.add_argument(
        "--results-file",
        default=os.environ.get("OUT", ""),
        help="Output results TSV path (default: env OUT).",
    )
    parser.add_argument(
        "--resolved-contracts-file",
        default=os.environ.get("LEC_RESOLVED_CONTRACTS_OUT", ""),
        help="Optional output TSV path for resolved per-case contract rows.",
    )
    parser.add_argument(
        "--drop-remark-cases-file",
        default=os.environ.get("LEC_DROP_REMARK_CASES_OUT", ""),
        help="Optional output TSV path for dropped-syntax case IDs.",
    )
    parser.add_argument(
        "--drop-remark-reasons-file",
        default=os.environ.get("LEC_DROP_REMARK_REASONS_OUT", ""),
        help="Optional output TSV path for dropped-syntax case+reason rows.",
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

    rule_filter_re: re.Pattern[str] | None = None
    if args.rule_filter:
        try:
            rule_filter_re = re.compile(args.rule_filter)
        except re.error as exc:
            fail(f"invalid --rule-filter: {args.rule_filter}: {exc}")

    target = parse_target_manifest(target_manifest)
    rules = parse_rules_manifest(rules_manifest)
    connection_groups = build_connection_groups(rules)
    filtered_groups = [
        group
        for group in connection_groups
        if group_matches_rule_filter(group, rule_filter_re)
    ]
    selected_groups = [
        group
        for idx, group in enumerate(filtered_groups)
        if (idx % rule_shard_count) == rule_shard_index
    ]

    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text("", encoding="utf-8")
    if not selected_groups:
        print(
            "No OpenTitan connectivity LEC cases selected.",
            file=sys.stderr,
            flush=True,
        )
        return 0

    if shutil.which(args.fusesoc_bin) is None and not Path(args.fusesoc_bin).exists():
        fail(f"fusesoc executable not found: {args.fusesoc_bin}")

    circt_verilog = os.environ.get("CIRCT_VERILOG", "build/bin/circt-verilog")
    circt_verilog_args = shlex.split(os.environ.get("CIRCT_VERILOG_ARGS", ""))
    circt_opt = os.environ.get("CIRCT_OPT", "build/bin/circt-opt")
    circt_opt_args = shlex.split(os.environ.get("CIRCT_OPT_ARGS", ""))
    circt_lec = os.environ.get("CIRCT_LEC", "build/bin/circt-lec")
    circt_lec_args = shlex.split(os.environ.get("CIRCT_LEC_ARGS", ""))
    timeout_secs = parse_nonnegative_int(
        os.environ.get("CIRCT_TIMEOUT_SECS", "300"), "CIRCT_TIMEOUT_SECS"
    )
    lec_run_smtlib = os.environ.get("LEC_RUN_SMTLIB", "1") == "1"
    lec_smoke_only = os.environ.get("LEC_SMOKE_ONLY", "0") == "1"
    lec_x_optimistic = os.environ.get("LEC_X_OPTIMISTIC", "0") == "1"
    lec_assume_known_inputs = os.environ.get("LEC_ASSUME_KNOWN_INPUTS", "0") == "1"
    lec_diagnose_xprop = os.environ.get("LEC_DIAGNOSE_XPROP", "0") == "1"
    lec_dump_unknown_sources = os.environ.get("LEC_DUMP_UNKNOWN_SOURCES", "0") == "1"
    lec_accept_xprop_only = os.environ.get("LEC_ACCEPT_XPROP_ONLY", "0") == "1"
    drop_remark_pattern = os.environ.get(
        "LEC_DROP_REMARK_PATTERN",
        os.environ.get("DROP_REMARK_PATTERN", "will be dropped during lowering"),
    )
    z3_bin = os.environ.get("Z3_BIN", "")

    if lec_x_optimistic and "--x-optimistic" not in circt_lec_args:
        circt_lec_args.append("--x-optimistic")
    if lec_assume_known_inputs and "--assume-known-inputs" not in circt_lec_args:
        circt_lec_args.append("--assume-known-inputs")
    if lec_diagnose_xprop and "--diagnose-xprop" not in circt_lec_args:
        circt_lec_args.append("--diagnose-xprop")
    if lec_dump_unknown_sources and "--dump-unknown-sources" not in circt_lec_args:
        circt_lec_args.append("--dump-unknown-sources")

    if lec_run_smtlib and not lec_smoke_only:
        if not z3_bin:
            z3_bin = shutil.which("z3") or ""
        if not z3_bin and Path.home().joinpath("z3-install/bin/z3").is_file():
            z3_bin = str(Path.home() / "z3-install/bin/z3")
        if not z3_bin and Path.home().joinpath("z3/build/z3").is_file():
            z3_bin = str(Path.home() / "z3/build/z3")
        if not z3_bin:
            fail("z3 not found; set Z3_BIN or disable LEC_RUN_SMTLIB")

    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="opentitan-conn-lec-")
        workdir = Path(temp_dir_obj.name)

    drop_remark_case_rows: list[tuple[str, str]] = []
    drop_remark_reason_rows: list[tuple[str, str, str]] = []
    drop_remark_seen_cases: set[str] = set()
    drop_remark_seen_case_reasons: set[tuple[str, str]] = set()
    resolved_contract_rows: list[tuple[str, ...]] = []
    rows: list[tuple[str, str, str, str, str, str]] = []

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

        checks_dir = workdir / "checks"
        checks_dir.mkdir(parents=True, exist_ok=True)
        cases: list[ConnectivityLECCase] = []
        skipped_connections = 0
        for index, group in enumerate(selected_groups):
            rule = group.connection
            src_expr_raw = block_signal_expr(rule.src_block, rule.src_signal, top_module)
            dst_expr_raw = block_signal_expr(rule.dest_block, rule.dest_signal, top_module)
            if not src_expr_raw or not dst_expr_raw:
                skipped_connections += 1
                continue
            src_expr = scope_under_instance(src_expr_raw, "dut")
            dst_expr = scope_under_instance(dst_expr_raw, "dut")
            guard_expr = build_condition_guard_expr(group.conditions, top_module, "dut")
            module_token = sanitize_token(f"conn_rule_{index}_{rule.rule_name}")
            if not module_token:
                module_token = f"conn_rule_{index}"
            ref_module = f"__circt_{module_token}_ref"
            impl_module = f"__circt_{module_token}_impl"
            checker_sv = checks_dir / f"__circt_{module_token}.sv"
            synthesize_rule_wrappers(
                checker_sv,
                ref_module,
                impl_module,
                top_module,
                src_expr,
                dst_expr,
                rule.rule_id,
                guard_expr,
            )
            cases.append(
                ConnectivityLECCase(
                    case_id=f"connectivity::{rule.rule_id}",
                    case_path=f"{rule.csv_file}:{rule.csv_row}",
                    rule_id=rule.rule_id,
                    checker_sv=checker_sv,
                    ref_module=ref_module,
                    impl_module=impl_module,
                )
            )

        if not cases:
            print(
                "No OpenTitan connectivity LEC cases selected.",
                file=sys.stderr,
                flush=True,
            )
            return 0

        contract_source = "manifest"
        contract_backend_mode = "smoke"
        if not lec_smoke_only:
            contract_backend_mode = "smtlib" if lec_run_smtlib else "jit"
        contract_z3_path = z3_bin if contract_backend_mode == "smtlib" else ""
        contract_lec_args = shlex.join(circt_lec_args)

        print(
            "opentitan connectivity lec: "
            f"target={target.target_name} selected_connections={len(selected_groups)} "
            f"selected_conditions={sum(len(group.conditions) for group in selected_groups)} "
            f"generated_cases={len(cases)} skipped_connections={skipped_connections} "
            f"top={top_module} shard={rule_shard_index}/{rule_shard_count}",
            file=sys.stderr,
            flush=True,
        )

        for case in cases:
            case_dir = workdir / "cases" / sanitize_token(case.case_id)
            case_dir.mkdir(parents=True, exist_ok=True)
            verilog_log = case_dir / "circt-verilog.log"
            opt_log = case_dir / "circt-opt.log"
            lec_log = case_dir / "circt-lec.log"
            lec_out = case_dir / "circt-lec.out"
            moore_mlir = case_dir / "connectivity.moore.mlir"
            core_mlir = case_dir / "connectivity.core.mlir"

            contract_fields = [
                contract_source,
                contract_backend_mode,
                args.mode_label,
                str(timeout_secs),
                "1" if lec_x_optimistic else "0",
                "1" if lec_assume_known_inputs else "0",
                "1" if lec_accept_xprop_only else "0",
                contract_z3_path,
                contract_lec_args,
            ]
            contract_fingerprint = compute_contract_fingerprint(contract_fields)
            resolved_contract_rows.append(
                (
                    case.case_id,
                    case.case_path,
                    *contract_fields,
                    contract_fingerprint,
                )
            )

            verilog_cmd = [
                circt_verilog,
                "--ir-moore",
                "-o",
                str(moore_mlir),
                "--single-unit",
                "--no-uvm-auto-include",
                f"--top={case.ref_module}",
                f"--top={case.impl_module}",
            ]
            for include_dir in include_dirs:
                verilog_cmd += ["-I", include_dir]
            verilog_cmd += circt_verilog_args
            verilog_cmd += source_files + [str(case.checker_sv)]

            opt_cmd = [
                circt_opt,
                str(moore_mlir),
                "--convert-moore-to-core",
                "--mlir-disable-threading",
                "-o",
                str(core_mlir),
            ]
            opt_cmd += circt_opt_args

            lec_cmd = [
                circt_lec,
                str(core_mlir),
                f"-c1={case.ref_module}",
                f"-c2={case.impl_module}",
            ]
            if lec_smoke_only:
                lec_cmd.append("--emit-mlir")
            elif lec_run_smtlib:
                lec_cmd.append("--run-smtlib")
                lec_cmd.append(f"--z3-path={z3_bin}")
            lec_cmd += circt_lec_args

            stage = "verilog"
            try:
                run_and_log(verilog_cmd, verilog_log, timeout_secs)
                if verilog_log.exists():
                    reasons = extract_drop_reasons(
                        verilog_log.read_text(encoding="utf-8"), drop_remark_pattern
                    )
                    if reasons and case.case_id not in drop_remark_seen_cases:
                        drop_remark_seen_cases.add(case.case_id)
                        drop_remark_case_rows.append((case.case_id, case.case_path))
                    for reason in reasons:
                        key = (case.case_id, reason)
                        if key in drop_remark_seen_case_reasons:
                            continue
                        drop_remark_seen_case_reasons.add(key)
                        drop_remark_reason_rows.append(
                            (case.case_id, case.case_path, reason)
                        )
                stage = "opt"
                run_and_log(opt_cmd, opt_log, timeout_secs)
                stage = "lec"
                combined = run_and_log(
                    lec_cmd, lec_log, timeout_secs, out_path=lec_out
                )
                diag = parse_lec_diag(combined)
                result = parse_lec_result(combined)
                if result in {"NEQ", "UNKNOWN"}:
                    if diag == "XPROP_ONLY" and lec_accept_xprop_only:
                        rows.append(
                            (
                                "XFAIL",
                                case.case_id,
                                case.case_path,
                                "opentitan",
                                args.mode_label,
                                "XPROP_ONLY",
                            )
                        )
                    else:
                        rows.append(
                            (
                                "FAIL",
                                case.case_id,
                                case.case_path,
                                "opentitan",
                                args.mode_label,
                                diag or result,
                            )
                        )
                elif result == "EQ":
                    rows.append(
                        (
                            "PASS",
                            case.case_id,
                            case.case_path,
                            "opentitan",
                            args.mode_label,
                            diag or "EQ",
                        )
                    )
                elif lec_smoke_only:
                    rows.append(
                        (
                            "PASS",
                            case.case_id,
                            case.case_path,
                            "opentitan",
                            args.mode_label,
                            diag or "SMOKE_ONLY",
                        )
                    )
                else:
                    rows.append(
                        (
                            "ERROR",
                            case.case_id,
                            case.case_path,
                            "opentitan",
                            args.mode_label,
                            diag or "CIRCT_LEC_ERROR",
                        )
                    )
            except subprocess.TimeoutExpired:
                if stage == "verilog":
                    diag = "CIRCT_VERILOG_TIMEOUT"
                elif stage == "opt":
                    diag = "CIRCT_OPT_TIMEOUT"
                else:
                    diag = "CIRCT_LEC_TIMEOUT"
                rows.append(
                    (
                        "TIMEOUT",
                        case.case_id,
                        case.case_path,
                        "opentitan",
                        args.mode_label,
                        diag,
                    )
                )
            except subprocess.CalledProcessError:
                if stage == "verilog":
                    rows.append(
                        (
                            "ERROR",
                            case.case_id,
                            case.case_path,
                            "opentitan",
                            args.mode_label,
                            "CIRCT_VERILOG_ERROR",
                        )
                    )
                elif stage == "opt":
                    rows.append(
                        (
                            "ERROR",
                            case.case_id,
                            case.case_path,
                            "opentitan",
                            args.mode_label,
                            "CIRCT_OPT_ERROR",
                        )
                    )
                else:
                    combined = ""
                    if lec_log.exists():
                        combined += lec_log.read_text(encoding="utf-8")
                    if lec_out.exists():
                        combined += "\n" + lec_out.read_text(encoding="utf-8")
                    diag = parse_lec_diag(combined)
                    result = parse_lec_result(combined)
                    if result in {"NEQ", "UNKNOWN"}:
                        if diag == "XPROP_ONLY" and lec_accept_xprop_only:
                            rows.append(
                                (
                                    "XFAIL",
                                    case.case_id,
                                    case.case_path,
                                    "opentitan",
                                    args.mode_label,
                                    "XPROP_ONLY",
                                )
                            )
                        else:
                            rows.append(
                                (
                                    "FAIL",
                                    case.case_id,
                                    case.case_path,
                                    "opentitan",
                                    args.mode_label,
                                    diag or result,
                                )
                            )
                    else:
                        rows.append(
                            (
                                "ERROR",
                                case.case_id,
                                case.case_path,
                                "opentitan",
                                args.mode_label,
                                diag or "CIRCT_LEC_ERROR",
                            )
                        )

        with results_file.open("w", encoding="utf-8") as handle:
            for row in sorted(rows, key=lambda item: (item[1], item[0], item[2])):
                handle.write("\t".join(row) + "\n")

        if args.drop_remark_cases_file:
            drop_case_path = Path(args.drop_remark_cases_file).resolve()
            drop_case_path.parent.mkdir(parents=True, exist_ok=True)
            with drop_case_path.open("w", encoding="utf-8") as handle:
                for row in sorted(drop_remark_case_rows, key=lambda item: item[0]):
                    handle.write("\t".join(row) + "\n")
        if args.drop_remark_reasons_file:
            drop_reason_path = Path(args.drop_remark_reasons_file).resolve()
            drop_reason_path.parent.mkdir(parents=True, exist_ok=True)
            with drop_reason_path.open("w", encoding="utf-8") as handle:
                for row in sorted(drop_remark_reason_rows, key=lambda item: (item[0], item[2])):
                    handle.write("\t".join(row) + "\n")
        if args.resolved_contracts_file:
            contracts_path = Path(args.resolved_contracts_file).resolve()
            contracts_path.parent.mkdir(parents=True, exist_ok=True)
            with contracts_path.open("w", encoding="utf-8") as handle:
                handle.write("#resolved_contract_schema_version=1\n")
                for row in sorted(
                    resolved_contract_rows, key=lambda item: (item[0], item[1])
                ):
                    handle.write("\t".join(row) + "\n")

        counts = {
            "total": 0,
            "pass": 0,
            "fail": 0,
            "xfail": 0,
            "xpass": 0,
            "error": 0,
            "skip": 0,
        }
        for row in rows:
            status = row[0].strip().upper()
            counts["total"] += 1
            if status == "PASS":
                counts["pass"] += 1
            elif status == "FAIL":
                counts["fail"] += 1
            elif status == "XFAIL":
                counts["xfail"] += 1
            elif status == "XPASS":
                counts["xpass"] += 1
            elif status == "SKIP":
                counts["skip"] += 1
            else:
                counts["error"] += 1
        print(
            "opentitan connectivity LEC summary: "
            f"total={counts['total']} pass={counts['pass']} fail={counts['fail']} "
            f"xfail={counts['xfail']} xpass={counts['xpass']} "
            f"error={counts['error']} skip={counts['skip']}",
            file=sys.stderr,
            flush=True,
        )
        return 0 if counts["fail"] == 0 and counts["error"] == 0 else 1
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

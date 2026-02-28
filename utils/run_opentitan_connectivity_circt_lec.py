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
from typing import Any, Sequence

import yaml

_THIS_DIR = Path(__file__).resolve().parent
_FORMAL_LIB_DIR = _THIS_DIR / "formal" / "lib"
if _FORMAL_LIB_DIR.is_dir():
    sys.path.insert(0, str(_FORMAL_LIB_DIR))


@dataclass(frozen=True)
class ConnectivityTarget:
    target_name: str
    fusesoc_core: str
    rel_path: str


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
    bind_top: str


CONNECTIVITY_LEC_STATUS_FIELDS = (
    "case_total",
    "case_pass",
    "case_fail",
    "case_xfail",
    "case_xpass",
    "case_error",
    "case_skip",
)


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


def parse_nonnegative_int_list(raw: str, name: str) -> list[int]:
    tokenized = raw.strip()
    if not tokenized:
        return []
    values: list[int] = []
    for index, token in enumerate(tokenized.split(","), start=1):
        item = token.strip()
        if not item:
            fail(f"invalid {name}: empty item at index {index}")
        values.append(parse_nonnegative_int(item, f"{name}[{index}]"))
    return values


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


def normalize_connectivity_rule_id(case_id: str) -> str:
    token = case_id.strip()
    prefix = "connectivity::"
    if token.startswith(prefix):
        return token[len(prefix) :]
    return token


def init_connectivity_lec_status_counts() -> dict[str, int]:
    return {field: 0 for field in CONNECTIVITY_LEC_STATUS_FIELDS}


def collect_connectivity_lec_status_counts(
    case_rows: list[tuple[str, ...]],
) -> dict[str, dict[str, int]]:
    by_rule: dict[str, dict[str, int]] = {}

    def get_counts(rule_id: str) -> dict[str, int]:
        return by_rule.setdefault(rule_id, init_connectivity_lec_status_counts())

    for row in case_rows:
        if len(row) < 2:
            continue
        rule_id = normalize_connectivity_rule_id(row[1])
        if not rule_id:
            continue
        status = (row[0] if row else "").strip().upper()
        counts = get_counts(rule_id)
        counts["case_total"] += 1
        if status == "PASS":
            counts["case_pass"] += 1
        elif status == "FAIL":
            counts["case_fail"] += 1
        elif status == "XFAIL":
            counts["case_xfail"] += 1
        elif status == "XPASS":
            counts["case_xpass"] += 1
        elif status == "SKIP":
            counts["case_skip"] += 1
        else:
            counts["case_error"] += 1

    return by_rule


def write_connectivity_lec_status_summary(
    path: Path,
    by_rule: dict[str, dict[str, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["rule_id", *CONNECTIVITY_LEC_STATUS_FIELDS])
        for rule_id in sorted(by_rule.keys()):
            counts = by_rule[rule_id]
            writer.writerow(
                [rule_id, *[counts[field] for field in CONNECTIVITY_LEC_STATUS_FIELDS]]
            )


def read_connectivity_lec_status_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        fail(f"connectivity LEC status summary file not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"connectivity LEC status summary missing header row: {path}")
        required = {"rule_id", *CONNECTIVITY_LEC_STATUS_FIELDS}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            fail(
                f"connectivity LEC status summary missing required columns {missing}: "
                f"{path} (found: {reader.fieldnames})"
            )
        out: dict[str, dict[str, str]] = {}
        for idx, row in enumerate(reader, start=2):
            rule_id = (row.get("rule_id") or "").strip()
            if not rule_id:
                continue
            if rule_id in out:
                fail(f"duplicate rule_id '{rule_id}' in {path} row {idx}")
            out[rule_id] = {
                field: (row.get(field) or "").strip()
                for field in CONNECTIVITY_LEC_STATUS_FIELDS
            }
    return out


def write_connectivity_lec_status_drift(
    path: Path,
    rows: list[tuple[str, str, str, str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["rule_id", "kind", "baseline", "current", "allowlisted"])
        for row in rows:
            writer.writerow(row)


def append_status_drift_error_row(
    results_path: Path,
    mode_label: str,
    baseline_path: Path,
) -> None:
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "ERROR\tconnectivity_status_drift\t"
            f"{baseline_path}\topentitan\t{mode_label}\t"
            "LEC_DRIFT_ERROR\tconnectivity_status_drift\n"
        )


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
    rel_path = (row.get("rel_path") or "").strip()
    return ConnectivityTarget(
        target_name=target_name,
        fusesoc_core=fusesoc_core,
        rel_path=rel_path,
    )


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


VERILOG_SOURCE_SUFFIXES = {".sv", ".v"}
VERILOG_HEADER_SUFFIXES = {".svh", ".vh"}


def classify_verilog_entry(file_type: Any, path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VERILOG_SOURCE_SUFFIXES:
        return "source"
    if suffix in VERILOG_HEADER_SUFFIXES:
        return "header"
    token = str(file_type or "").strip().lower()
    if "verilog" not in token:
        return ""
    return "source"


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
        verilog_kind = classify_verilog_entry(entry.get("file_type"), file_path)
        if not verilog_kind:
            continue
        if entry.get("is_include_file"):
            incdir = str(file_path.parent)
            if incdir not in seen_incdirs:
                include_dirs.append(incdir)
                seen_incdirs.add(incdir)
            # Keep include-marked Verilog source files in the compile unit.
            # OpenTitan formal targets rely on some `.sv` macro libraries
            # (for example `prim_assert.sv`) being compiled, while header-only
            # include files (`.svh`, `.vh`) should remain include-only.
            if verilog_kind != "source":
                continue
        elif verilog_kind == "header":
            incdir = str(file_path.parent)
            if incdir not in seen_incdirs:
                include_dirs.append(incdir)
                seen_incdirs.add(incdir)
            continue
        files.append(file_text)
    return files, include_dirs


def apply_platform_file_filter(rel_path: str, files: list[str]) -> list[str]:
    rel = rel_path.lower()
    if "top_earlgrey" in rel:
        return [item for item in files if "/lowrisc_englishbreakfast_" not in item]
    if "top_englishbreakfast" in rel:
        return [item for item in files if "/lowrisc_earlgrey_" not in item]
    return files


def apply_prim_impl_file_filter(target_name: str, files: list[str]) -> list[str]:
    target = target_name.lower()
    if "asic" in target:
        return [item for item in files if "/lowrisc_prim_xilinx_" not in item]
    return files


def infer_rule_top_override(groups: Sequence[ConnectivityConnectionGroup]) -> str:
    tops: set[str] = set()
    for group in groups:
        for rule in (group.connection, *group.conditions):
            block = rule.src_block.strip() or rule.dest_block.strip()
            if not block:
                continue
            top = block.split(".", 1)[0].strip()
            if top:
                tops.add(top)
    if len(tops) != 1:
        return ""
    return next(iter(tops))


def apply_top_override_source_prune(
    source_files: list[str], current_top: str, override_top: str
) -> list[str]:
    if not current_top or not override_top or current_top == override_top:
        return source_files
    suffixes = (f"/{current_top}.sv", f"/{current_top}.v")
    return [path for path in source_files if not path.endswith(suffixes)]


def infer_external_hierarchy_top_fallback(
    groups: Sequence[ConnectivityConnectionGroup],
    current_top: str,
    target_name: str,
    source_files: Sequence[str],
) -> str:
    """Select a chip-wrapper top when rules reference external sibling blocks.

    OpenTitan connectivity rows may mix `top_*.*` paths with sibling instances
    like `u_ast.*`. If FuseSoC exposes `top_*` as toplevel, those sibling
    instances are unreachable. In that case, fall back to the target chip
    wrapper module when it is present in the source set.
    """

    if not current_top or not target_name or current_top == target_name:
        return ""
    roots: set[str] = set()
    for group in groups:
        for block in (group.connection.src_block, group.connection.dest_block):
            token = block.strip()
            if not token:
                continue
            root = token.split(".", 1)[0].strip()
            if root:
                roots.add(root)
    if not roots or all(root == current_top for root in roots):
        return ""
    target_suffixes = (f"/{target_name}.sv", f"/{target_name}.v")
    if not any(path.endswith(target_suffixes) for path in source_files):
        return ""
    return target_name


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


CONST_INDEXED_BIT_RE = re.compile(r"^\s*(.+?)\[(\d+)\]\s*$")


def rewrite_const_indexed_bit(expr: str) -> str:
    """Rewrite `foo[3]` to shift/mask form to avoid frontend crashes.

    Some OpenTitan connectivity paths trigger `circt-verilog` crashes in
    SimplifyProcedures when directly using hierarchical constant bit-select
    syntax in generated wrappers. Rewrite the common constant-index case into
    an equivalent `$unsigned` shift/mask form.
    """

    match = CONST_INDEXED_BIT_RE.fullmatch(expr.strip())
    if not match:
        return expr
    base_expr = match.group(1).strip()
    index = match.group(2)
    return f"((($unsigned({base_expr})) >> {index}) & 1'b1)"


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
        cond_expr = rewrite_const_indexed_bit(
            scope_under_instance(cond_expr_raw, instance_name)
        )
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


def _coerce_text(payload: str | bytes | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return payload


def write_log(path: Path, stdout: str | bytes | None, stderr: str | bytes | None) -> None:
    stdout_text = _coerce_text(stdout)
    stderr_text = _coerce_text(stderr)
    data = ""
    if stdout_text:
        data += stdout_text
        if not data.endswith("\n"):
            data += "\n"
    if stderr_text:
        data += stderr_text
    path.write_text(data, encoding="utf-8")


def strip_vpi_attributes_for_opt(path: Path) -> bool:
    """Strip trailing vpi.* op attributes that circt-opt may fail to parse."""
    text = path.read_text(encoding="utf-8")
    had_trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    changed = False
    stripped: list[str] = []
    for line in lines:
        marker = " attributes {vpi."
        idx = line.find(marker)
        if idx >= 0:
            stripped.append(line[:idx])
            changed = True
            continue
        stripped.append(line)
    if not changed:
        return False
    out = "\n".join(stripped)
    if had_trailing_newline:
        out += "\n"
    path.write_text(out, encoding="utf-8")
    return True


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


def is_llhd_abstraction_unknown(result: str | None, diag: str | None) -> bool:
    return result == "UNKNOWN" and diag == "LLHD_ABSTRACTION"


def has_command_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def has_preprocessor_define(args: Sequence[str], macro: str) -> bool:
    index = 0
    while index < len(args):
        arg = args[index]
        payload = ""
        step = 1
        if arg == "-D":
            if index + 1 >= len(args):
                break
            payload = args[index + 1]
            step = 2
        elif arg.startswith("-D"):
            payload = arg[2:]
        if payload:
            token = payload.strip()
            if token == macro or token.startswith(f"{macro}="):
                return True
        index += step
    return False


def is_missing_timescale_retryable_failure(log_text: str) -> bool:
    low = log_text.lower()
    return (
        "design element does not have a time scale defined but others in the design do"
        in low
    )


def is_always_comb_multi_driver_retryable_failure(log_text: str) -> bool:
    return "driven by always_comb procedure" in log_text.lower()


def is_resource_guard_rss_retryable_failure(log_text: str) -> bool:
    low = log_text.lower()
    return "resource guard triggered: rss" in low and "exceeded limit" in low


def has_explicit_resource_guard_policy(args: list[str]) -> bool:
    explicit_limit_options = (
        "--max-rss-mb",
        "--max-vmem-mb",
        "--max-malloc-mb",
        "--max-wall-ms",
    )
    if has_command_option(args, "--no-resource-guard"):
        return True
    if any(has_command_option(args, option) for option in explicit_limit_options):
        return True
    explicit_env_vars = (
        "CIRCT_MAX_RSS_MB",
        "CIRCT_MAX_VMEM_MB",
        "CIRCT_MAX_MALLOC_MB",
        "CIRCT_MAX_WALL_MS",
    )
    return any(os.environ.get(name, "").strip() for name in explicit_env_vars)


def is_temporal_approx_retryable_failure(log_text: str) -> bool:
    low = log_text.lower()
    return (
        "ltl.delay with delay > 0 must be lowered by the bmc multi-step infrastructure"
        in low
    )


def is_opt_emit_bytecode_retryable_failure(log_text: str) -> bool:
    low = log_text.lower()
    return (
        "--emit-bytecode" in low
        and (
            "unknown command line argument" in low
            or "unknown argument" in low
        )
    )


def is_disable_threading_retryable_failure(returncode: int, log_text: str) -> bool:
    if returncode in {-11, -6, 134, 139}:
        return True
    low = log_text.lower()
    return (
        "segmentation fault" in low
        or "signal 11" in low
        or ("stack dump:" in low and "please submit a bug report" in low)
    )


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
        stdout = _coerce_text(exc.stdout)
        stderr = _coerce_text(exc.stderr)
        write_log(log_path, stdout, stderr)
        if out_path is not None:
            out_path.write_text(stdout, encoding="utf-8")
        raise
    except OSError as exc:
        err_text = f"{exc.__class__.__name__}: {exc}"
        write_log(log_path, "", err_text)
        if out_path is not None:
            out_path.write_text("", encoding="utf-8")
        raise subprocess.CalledProcessError(
            127, cmd, output="", stderr=err_text
        ) from exc
    stdout = _coerce_text(result.stdout)
    stderr = _coerce_text(result.stderr)
    write_log(log_path, stdout, stderr)
    if out_path is not None:
        out_path.write_text(stdout, encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    return stdout + "\n" + stderr


def compute_contract_fingerprint(fields: list[str]) -> str:
    payload = "\x1f".join(fields).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


try:
    from runner_common import (
        is_allowlisted as _shared_is_allowlisted,
        load_allowlist as _shared_load_allowlist,
        parse_exit_codes as _shared_parse_exit_codes,
        parse_nonnegative_float as _shared_parse_nonnegative_float,
        parse_nonnegative_int as _shared_parse_nonnegative_int,
        read_status_summary as _shared_read_status_summary,
        run_command_logged as _shared_run_command_logged,
        write_log as _shared_write_log,
        write_status_drift as _shared_write_status_drift,
        write_status_summary as _shared_write_status_summary,
    )
except Exception:
    _HAS_SHARED_FORMAL_HELPERS = False
else:
    _HAS_SHARED_FORMAL_HELPERS = True

if _HAS_SHARED_FORMAL_HELPERS:

    def parse_nonnegative_int(raw: str, name: str) -> int:
        return _shared_parse_nonnegative_int(raw, name, fail)

    def load_allowlist(path: Path) -> tuple[set[str], list[str], list[re.Pattern[str]]]:
        return _shared_load_allowlist(path, fail)

    def is_allowlisted(
        token: str, exact: set[str], prefixes: list[str], regex_rules: list[re.Pattern[str]]
    ) -> bool:
        return _shared_is_allowlisted(token, (exact, prefixes, regex_rules))

    def write_connectivity_lec_status_summary(
        path: Path,
        by_rule: dict[str, dict[str, int]],
    ) -> None:
        _shared_write_status_summary(path, CONNECTIVITY_LEC_STATUS_FIELDS, by_rule)

    def read_connectivity_lec_status_summary(path: Path) -> dict[str, dict[str, str]]:
        return _shared_read_status_summary(
            path,
            CONNECTIVITY_LEC_STATUS_FIELDS,
            "connectivity LEC status summary",
            fail,
        )

    def write_connectivity_lec_status_drift(
        path: Path,
        rows: list[tuple[str, str, str, str, str]],
    ) -> None:
        _shared_write_status_drift(path, rows)

    def write_log(path: Path, stdout: str, stderr: str) -> None:
        _shared_write_log(path, stdout, stderr)

    def run_and_log(
        cmd: list[str],
        log_path: Path,
        timeout_secs: int,
        out_path: Path | None = None,
    ) -> str:
        retry_attempts = _shared_parse_nonnegative_int(
            os.environ.get("FORMAL_LAUNCH_RETRY_ATTEMPTS", "1"),
            "FORMAL_LAUNCH_RETRY_ATTEMPTS",
            fail,
        )
        retry_backoff_secs = _shared_parse_nonnegative_float(
            os.environ.get("FORMAL_LAUNCH_RETRY_BACKOFF_SECS", "0.2"),
            "FORMAL_LAUNCH_RETRY_BACKOFF_SECS",
            fail,
        )
        retryable_exit_codes = _shared_parse_exit_codes(
            os.environ.get("FORMAL_LAUNCH_RETRYABLE_EXIT_CODES", "126,127"),
            "FORMAL_LAUNCH_RETRYABLE_EXIT_CODES",
            fail,
        )
        retryable_patterns_raw = os.environ.get(
            "FORMAL_LAUNCH_RETRYABLE_PATTERNS",
            "text file busy,resource temporarily unavailable,stale file handle",
        )
        retryable_patterns = [
            token.strip()
            for token in retryable_patterns_raw.split(",")
            if token.strip()
        ]
        return _shared_run_command_logged(
            cmd,
            log_path,
            timeout_secs=timeout_secs,
            out_path=out_path,
            retry_attempts=retry_attempts,
            retry_backoff_secs=retry_backoff_secs,
            retryable_exit_codes=retryable_exit_codes,
            retryable_output_patterns=retryable_patterns,
        )


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
        "--status-summary-file",
        default=os.environ.get("LEC_CONNECTIVITY_STATUS_SUMMARY_OUT", ""),
        help=(
            "Optional output TSV path for per-rule connectivity LEC status "
            "counters."
        ),
    )
    parser.add_argument(
        "--status-baseline-file",
        default=os.environ.get("LEC_CONNECTIVITY_STATUS_BASELINE_FILE", ""),
        help=(
            "Optional baseline per-rule connectivity LEC status summary TSV used for "
            "drift checking."
        ),
    )
    parser.add_argument(
        "--status-drift-file",
        default=os.environ.get("LEC_CONNECTIVITY_STATUS_DRIFT_OUT", ""),
        help="Optional output TSV path for connectivity LEC status drift rows.",
    )
    parser.add_argument(
        "--status-drift-allowlist-file",
        default=os.environ.get("LEC_CONNECTIVITY_STATUS_DRIFT_ALLOWLIST_FILE", ""),
        help=(
            "Optional allowlist file for connectivity LEC status drift suppression "
            "(rule_id exact/prefix/regex)."
        ),
    )
    parser.add_argument(
        "--fail-on-status-drift",
        action="store_true",
        default=os.environ.get("LEC_FAIL_ON_CONNECTIVITY_STATUS_DRIFT", "0") == "1",
        help="Fail when connectivity LEC status drift is detected vs baseline.",
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
    parser.add_argument(
        "--timeout-reasons-file",
        default=os.environ.get("LEC_TIMEOUT_REASON_CASES_OUT", ""),
        help="Optional output TSV path for timeout reason rows.",
    )
    args = parser.parse_args()

    if not args.results_file:
        fail("missing --results-file (or OUT environment)")

    target_manifest = Path(args.target_manifest).resolve()
    rules_manifest = Path(args.rules_manifest).resolve()
    opentitan_root = Path(args.opentitan_root).resolve()
    results_file = Path(args.results_file).resolve()
    status_summary_path = (
        Path(args.status_summary_file).resolve() if args.status_summary_file else None
    )
    status_baseline_path = (
        Path(args.status_baseline_file).resolve() if args.status_baseline_file else None
    )
    status_drift_path = (
        Path(args.status_drift_file).resolve() if args.status_drift_file else None
    )
    status_allowlist_path = (
        Path(args.status_drift_allowlist_file).resolve()
        if args.status_drift_allowlist_file
        else None
    )
    if not opentitan_root.is_dir():
        fail(f"opentitan root not found: {opentitan_root}")
    if status_baseline_path is not None and not status_baseline_path.is_file():
        fail(f"connectivity LEC status baseline file not found: {status_baseline_path}")
    if status_allowlist_path is not None and not status_allowlist_path.is_file():
        fail(
            "connectivity LEC status drift allowlist file not found: "
            f"{status_allowlist_path}"
        )

    rule_shard_count = parse_nonnegative_int(args.rule_shard_count, "rule-shard-count")
    rule_shard_index = parse_nonnegative_int(args.rule_shard_index, "rule-shard-index")
    if rule_shard_count < 1:
        fail("invalid --rule-shard-count: expected integer >= 1")
    if rule_shard_index >= rule_shard_count:
        fail("invalid --rule-shard-index: expected value < --rule-shard-count")
    if args.fail_on_status_drift and not args.status_baseline_file:
        fail("--fail-on-status-drift requires --status-baseline-file")
    if args.status_drift_allowlist_file and not args.status_baseline_file:
        fail("--status-drift-allowlist-file requires --status-baseline-file")

    allow_exact: set[str] = set()
    allow_prefix: list[str] = []
    allow_regex: list[re.Pattern[str]] = []
    if status_allowlist_path is not None:
        allow_exact, allow_prefix, allow_regex = load_allowlist(status_allowlist_path)

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

    def evaluate_status_governance(case_rows: list[tuple[str, ...]]) -> int:
        current_counts = collect_connectivity_lec_status_counts(case_rows)
        if status_summary_path is not None:
            write_connectivity_lec_status_summary(status_summary_path, current_counts)
            print(f"connectivity LEC status summary: {status_summary_path}", flush=True)
        if status_baseline_path is None:
            return 0

        baseline = read_connectivity_lec_status_summary(status_baseline_path)
        current = {
            rule_id: {
                field: str(counts[field]) for field in CONNECTIVITY_LEC_STATUS_FIELDS
            }
            for rule_id, counts in current_counts.items()
        }
        drift_rows: list[tuple[str, str, str, str, str]] = []
        non_allowlisted_rows: list[tuple[str, str, str, str, str]] = []
        baseline_rules = set(baseline.keys())
        current_rules = set(current.keys())

        def add_drift(rule_id: str, kind: str, before: str, after: str) -> None:
            allowlisted = is_allowlisted(rule_id, allow_exact, allow_prefix, allow_regex)
            row = (rule_id, kind, before, after, "1" if allowlisted else "0")
            drift_rows.append(row)
            if not allowlisted:
                non_allowlisted_rows.append(row)

        for rule_id in sorted(baseline_rules - current_rules):
            add_drift(rule_id, "missing_in_current", "present", "absent")
        for rule_id in sorted(current_rules - baseline_rules):
            add_drift(rule_id, "new_in_current", "absent", "present")
        for rule_id in sorted(baseline_rules.intersection(current_rules)):
            before_row = baseline[rule_id]
            after_row = current[rule_id]
            for kind in CONNECTIVITY_LEC_STATUS_FIELDS:
                before = before_row.get(kind, "")
                after = after_row.get(kind, "")
                if before != after:
                    add_drift(rule_id, kind, before, after)

        if status_drift_path is not None:
            write_connectivity_lec_status_drift(status_drift_path, drift_rows)
            print(f"connectivity LEC status drift: {status_drift_path}", flush=True)

        if non_allowlisted_rows:
            sample = ", ".join(
                f"{rule_id}:{kind}"
                for rule_id, kind, _, _, _ in non_allowlisted_rows[:6]
            )
            if len(non_allowlisted_rows) > 6:
                sample += ", ..."
            message = (
                "opentitan connectivity LEC status drift detected: "
                f"rows={len(non_allowlisted_rows)} sample=[{sample}] "
                f"baseline={status_baseline_path}"
            )
            if args.fail_on_status_drift:
                append_status_drift_error_row(
                    results_file, args.mode_label, status_baseline_path
                )
                print(message, file=sys.stderr)
                return 1
            print(f"warning: {message}", file=sys.stderr)
        else:
            print(
                "opentitan connectivity LEC status drift check passed: "
                f"rules={len(current)} baseline={status_baseline_path}",
                file=sys.stderr,
            )
        return 0

    if not selected_groups:
        if status_summary_path is not None:
            write_connectivity_lec_status_summary(status_summary_path, {})
            print(f"connectivity LEC status summary: {status_summary_path}", flush=True)
        if status_drift_path is not None:
            write_connectivity_lec_status_drift(status_drift_path, [])
            print(f"connectivity LEC status drift: {status_drift_path}", flush=True)
        print(
            "No OpenTitan connectivity LEC cases selected.",
            file=sys.stderr,
            flush=True,
        )
        return 0

    if shutil.which(args.fusesoc_bin) is None and not Path(args.fusesoc_bin).exists():
        fail(f"fusesoc executable not found: {args.fusesoc_bin}")

    circt_verilog = os.environ.get("CIRCT_VERILOG", "build_test/bin/circt-verilog")
    circt_verilog_args = shlex.split(os.environ.get("CIRCT_VERILOG_ARGS", ""))
    circt_opt = os.environ.get("CIRCT_OPT", "build_test/bin/circt-opt")
    circt_opt_args = shlex.split(os.environ.get("CIRCT_OPT_ARGS", ""))
    circt_lec = os.environ.get("CIRCT_LEC", "build_test/bin/circt-lec")
    circt_lec_args = shlex.split(os.environ.get("CIRCT_LEC_ARGS", ""))
    has_explicit_verify_each = has_command_option(circt_lec_args, "--verify-each")
    # Connectivity LEC cases can be substantially larger than AES S-Box parity
    # checks; keep a higher default command timeout to avoid spurious
    # infrastructure timeouts on valid long-running cases.
    timeout_secs = parse_nonnegative_int(
        os.environ.get("CIRCT_TIMEOUT_SECS", "600"), "CIRCT_TIMEOUT_SECS"
    )
    lec_run_smtlib = os.environ.get("LEC_RUN_SMTLIB", "1") == "1"
    lec_smoke_only = os.environ.get("LEC_SMOKE_ONLY", "0") == "1"
    lec_timeout_frontier_probe = (
        os.environ.get("LEC_TIMEOUT_FRONTIER_PROBE", "1") == "1"
    )
    lec_x_optimistic = os.environ.get("LEC_X_OPTIMISTIC", "0") == "1"
    lec_assume_known_inputs = os.environ.get("LEC_ASSUME_KNOWN_INPUTS", "0") == "1"
    lec_diagnose_xprop = os.environ.get("LEC_DIAGNOSE_XPROP", "0") == "1"
    lec_dump_unknown_sources = os.environ.get("LEC_DUMP_UNKNOWN_SOURCES", "0") == "1"
    lec_accept_xprop_only = os.environ.get("LEC_ACCEPT_XPROP_ONLY", "0") == "1"
    # OpenTitan connectivity parity accepts known LLHD abstraction diagnostics
    # by default; set LEC_ACCEPT_LLHD_ABSTRACTION=0 to disable.
    lec_accept_llhd_abstraction = (
        os.environ.get("LEC_ACCEPT_LLHD_ABSTRACTION", "1") == "1"
    )
    llhd_abstraction_assume_known_inputs_retry_mode = os.environ.get(
        "LEC_LLHD_ABSTRACTION_ASSUME_KNOWN_INPUTS_RETRY_MODE", "auto"
    ).strip().lower()
    if llhd_abstraction_assume_known_inputs_retry_mode not in {"auto", "on", "off"}:
        fail(
            (
                "invalid LEC_LLHD_ABSTRACTION_ASSUME_KNOWN_INPUTS_RETRY_MODE: "
                f"{llhd_abstraction_assume_known_inputs_retry_mode} "
                "(expected auto|on|off)"
            )
        )
    verilog_timescale_fallback_mode = os.environ.get(
        "LEC_VERILOG_TIMESCALE_FALLBACK_MODE", "auto"
    ).strip().lower()
    if verilog_timescale_fallback_mode not in {"auto", "on", "off"}:
        fail(
            (
                "invalid LEC_VERILOG_TIMESCALE_FALLBACK_MODE: "
                f"{verilog_timescale_fallback_mode} (expected auto|on|off)"
            )
        )
    verilog_fallback_timescale = os.environ.get(
        "LEC_VERILOG_FALLBACK_TIMESCALE", "1ns/1ps"
    ).strip()
    if verilog_timescale_fallback_mode != "off" and not verilog_fallback_timescale:
        fail(
            (
                "invalid LEC_VERILOG_FALLBACK_TIMESCALE: "
                "expected non-empty timescale value"
            )
        )
    verilog_always_comb_multi_driver_mode = os.environ.get(
        "LEC_VERILOG_ALWAYS_COMB_MULTI_DRIVER_MODE", "auto"
    ).strip().lower()
    if verilog_always_comb_multi_driver_mode not in {"auto", "on", "off"}:
        fail(
            (
                "invalid LEC_VERILOG_ALWAYS_COMB_MULTI_DRIVER_MODE: "
                f"{verilog_always_comb_multi_driver_mode} (expected auto|on|off)"
            )
        )
    verilog_auto_relax_resource_guard_raw = os.environ.get(
        "LEC_VERILOG_AUTO_RELAX_RESOURCE_GUARD", "1"
    )
    if verilog_auto_relax_resource_guard_raw not in {"0", "1"}:
        fail(
            (
                "invalid LEC_VERILOG_AUTO_RELAX_RESOURCE_GUARD: "
                f"{verilog_auto_relax_resource_guard_raw} (expected 0 or 1)"
            )
        )
    verilog_auto_relax_resource_guard = (
        verilog_auto_relax_resource_guard_raw == "1"
    )
    verilog_auto_relax_resource_guard_max_rss_mb = parse_nonnegative_int(
        os.environ.get("LEC_VERILOG_AUTO_RELAX_RESOURCE_GUARD_MAX_RSS_MB", "24576"),
        "LEC_VERILOG_AUTO_RELAX_RESOURCE_GUARD_MAX_RSS_MB",
    )
    verilog_auto_relax_resource_guard_rss_ladder_mb = parse_nonnegative_int_list(
        os.environ.get("LEC_VERILOG_AUTO_RELAX_RESOURCE_GUARD_RSS_LADDER_MB", ""),
        "LEC_VERILOG_AUTO_RELAX_RESOURCE_GUARD_RSS_LADDER_MB",
    )
    if (
        not verilog_auto_relax_resource_guard_rss_ladder_mb
        and verilog_auto_relax_resource_guard_max_rss_mb > 0
    ):
        verilog_auto_relax_resource_guard_rss_ladder_mb = [
            verilog_auto_relax_resource_guard_max_rss_mb
        ]
    verilog_auto_relax_resource_guard_rss_ladder_mb = sorted(
        {
            value
            for value in verilog_auto_relax_resource_guard_rss_ladder_mb
            if value > 0
        }
    )
    temporal_approx_mode = os.environ.get(
        "LEC_TEMPORAL_APPROX_MODE", "auto"
    ).strip().lower()
    if temporal_approx_mode not in {"auto", "on", "off"}:
        fail(
            (
                "invalid LEC_TEMPORAL_APPROX_MODE: "
                f"{temporal_approx_mode} (expected auto|on|off)"
            )
        )
    verify_each_mode = os.environ.get("LEC_VERIFY_EACH_MODE", "auto").strip().lower()
    if verify_each_mode not in {"auto", "on", "off"}:
        fail(
            (
                "invalid LEC_VERIFY_EACH_MODE: "
                f"{verify_each_mode} (expected auto|on|off)"
            )
        )
    disable_threading_retry_mode = os.environ.get(
        "LEC_DISABLE_THREADING_RETRY_MODE", "auto"
    ).strip().lower()
    if disable_threading_retry_mode not in {"auto", "on", "off"}:
        fail(
            (
                "invalid LEC_DISABLE_THREADING_RETRY_MODE: "
                f"{disable_threading_retry_mode} (expected auto|on|off)"
            )
        )
    canonicalizer_timeout_retry_mode = os.environ.get(
        "LEC_CANONICALIZER_TIMEOUT_RETRY_MODE", "auto"
    ).strip().lower()
    if canonicalizer_timeout_retry_mode not in {"auto", "on", "off"}:
        fail(
            (
                "invalid LEC_CANONICALIZER_TIMEOUT_RETRY_MODE: "
                f"{canonicalizer_timeout_retry_mode} (expected auto|on|off)"
            )
        )
    canonicalizer_timeout_retry_max_iterations = parse_nonnegative_int(
        os.environ.get("LEC_CANONICALIZER_TIMEOUT_RETRY_MAX_ITERATIONS", "0"),
        "LEC_CANONICALIZER_TIMEOUT_RETRY_MAX_ITERATIONS",
    )
    canonicalizer_timeout_retry_max_num_rewrites = parse_nonnegative_int(
        os.environ.get("LEC_CANONICALIZER_TIMEOUT_RETRY_MAX_NUM_REWRITES", "40000"),
        "LEC_CANONICALIZER_TIMEOUT_RETRY_MAX_NUM_REWRITES",
    )
    canonicalizer_timeout_retry_rewrite_ladder = parse_nonnegative_int_list(
        os.environ.get(
            "LEC_CANONICALIZER_TIMEOUT_RETRY_REWRITE_LADDER",
            "20000,10000,5000,2000,1000,500",
        ),
        "LEC_CANONICALIZER_TIMEOUT_RETRY_REWRITE_LADDER",
    )
    canonicalizer_timeout_retry_rewrite_ladder = sorted(
        {value for value in canonicalizer_timeout_retry_rewrite_ladder if value > 0},
        reverse=True,
    )
    canonicalizer_timeout_retry_timeout_secs = parse_nonnegative_int(
        os.environ.get("LEC_CANONICALIZER_TIMEOUT_RETRY_TIMEOUT_SECS", "180"),
        "LEC_CANONICALIZER_TIMEOUT_RETRY_TIMEOUT_SECS",
    )
    canonicalizer_timeout_retry_auto_preenable_timeout_secs = parse_nonnegative_int(
        os.environ.get(
            "LEC_CANONICALIZER_TIMEOUT_RETRY_AUTO_PREENABLE_TIMEOUT_SECS", "120"
        ),
        "LEC_CANONICALIZER_TIMEOUT_RETRY_AUTO_PREENABLE_TIMEOUT_SECS",
    )
    if canonicalizer_timeout_retry_mode != "off":
        if (
            canonicalizer_timeout_retry_max_iterations <= 0
            and canonicalizer_timeout_retry_max_num_rewrites <= 0
        ):
            fail(
                "invalid canonicalizer timeout retry budget: expected "
                "LEC_CANONICALIZER_TIMEOUT_RETRY_MAX_ITERATIONS > 0 or "
                "LEC_CANONICALIZER_TIMEOUT_RETRY_MAX_NUM_REWRITES > 0 "
                "when timeout retry mode is enabled"
            )
    opt_emit_bytecode_mode = os.environ.get(
        "LEC_OPT_EMIT_BYTECODE_MODE", "auto"
    ).strip().lower()
    if opt_emit_bytecode_mode not in {"auto", "on", "off"}:
        fail(
            (
                "invalid LEC_OPT_EMIT_BYTECODE_MODE: "
                f"{opt_emit_bytecode_mode} (expected auto|on|off)"
            )
        )
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
    if (
        lec_accept_llhd_abstraction
        and "--accept-llhd-abstraction" not in circt_lec_args
    ):
        circt_lec_args.append("--accept-llhd-abstraction")

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
    timeout_reason_rows: list[tuple[str, str, str]] = []
    timeout_reason_seen: set[tuple[str, str]] = set()
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
        default_top_module = toplevels[0]
        source_files, include_dirs = resolve_eda_paths(eda_obj.get("files"), eda_yml.parent)
        source_files = apply_platform_file_filter(target.rel_path, source_files)
        source_files = apply_prim_impl_file_filter(target.target_name, source_files)
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
        fallback_tops_used: set[str] = set()
        case_tops_used: set[str] = set()
        for index, group in enumerate(selected_groups):
            case_top_module = default_top_module
            rule_top_override = infer_rule_top_override((group,))
            if rule_top_override:
                case_top_module = rule_top_override
            hierarchy_top_fallback = infer_external_hierarchy_top_fallback(
                (group,), case_top_module, target.target_name, source_files
            )
            if hierarchy_top_fallback:
                fallback_tops_used.add(hierarchy_top_fallback)
                case_top_module = hierarchy_top_fallback
            case_tops_used.add(case_top_module)
            rule = group.connection
            src_expr_raw = block_signal_expr(
                rule.src_block, rule.src_signal, case_top_module
            )
            dst_expr_raw = block_signal_expr(
                rule.dest_block, rule.dest_signal, case_top_module
            )
            if not src_expr_raw or not dst_expr_raw:
                skipped_connections += 1
                continue
            src_expr = rewrite_const_indexed_bit(scope_under_instance(src_expr_raw, "dut"))
            dst_expr = rewrite_const_indexed_bit(scope_under_instance(dst_expr_raw, "dut"))
            guard_expr = build_condition_guard_expr(
                group.conditions, case_top_module, "dut"
            )
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
                case_top_module,
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
                    bind_top=case_top_module,
                )
            )

        if not cases:
            if status_summary_path is not None:
                write_connectivity_lec_status_summary(status_summary_path, {})
                print(f"connectivity LEC status summary: {status_summary_path}", flush=True)
            if status_drift_path is not None:
                write_connectivity_lec_status_drift(status_drift_path, [])
                print(f"connectivity LEC status drift: {status_drift_path}", flush=True)
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
        for fallback_top in sorted(fallback_tops_used):
            print(
                "opentitan connectivity lec: using chip-wrapper top fallback "
                f"{fallback_top} (eda_top={default_top_module})",
                file=sys.stderr,
                flush=True,
            )
        case_top_summary = (
            next(iter(case_tops_used)) if len(case_tops_used) == 1 else "mixed"
        )

        print(
            "opentitan connectivity lec: "
            f"target={target.target_name} selected_connections={len(selected_groups)} "
            f"selected_conditions={sum(len(group.conditions) for group in selected_groups)} "
            f"generated_cases={len(cases)} skipped_connections={skipped_connections} "
            f"top={case_top_summary} shard={rule_shard_index}/{rule_shard_count}",
            file=sys.stderr,
            flush=True,
        )

        contract_fields = [
            contract_source,
            contract_backend_mode,
            args.mode_label,
            str(timeout_secs),
            "1" if lec_x_optimistic else "0",
            "1" if lec_assume_known_inputs else "0",
            "1" if lec_accept_xprop_only else "0",
            verilog_timescale_fallback_mode,
            verilog_fallback_timescale if verilog_timescale_fallback_mode != "off" else "",
            temporal_approx_mode,
            contract_z3_path,
            contract_lec_args,
        ]
        contract_fingerprint = compute_contract_fingerprint(contract_fields)
        for case in cases:
            resolved_contract_rows.append(
                (
                    case.case_id,
                    case.case_path,
                    *contract_fields,
                    contract_fingerprint,
                )
            )

        has_explicit_timescale = has_command_option(circt_verilog_args, "--timescale")
        has_explicit_multi_driver = has_command_option(
            circt_verilog_args, "--allow-multi-always-comb-drivers"
        )
        has_explicit_verilog_resource_guard_policy = has_explicit_resource_guard_policy(
            circt_verilog_args
        )
        has_explicit_temporal_approx = has_command_option(
            circt_lec_args, "--approx-temporal"
        )
        has_explicit_disable_threading = has_command_option(
            circt_lec_args, "--mlir-disable-threading"
        )
        has_explicit_canonicalizer_max_iterations = has_command_option(
            circt_lec_args, "--lec-canonicalizer-max-iterations"
        )
        has_explicit_canonicalizer_max_num_rewrites = has_command_option(
            circt_lec_args, "--lec-canonicalizer-max-num-rewrites"
        )
        has_assume_known_inputs = has_command_option(
            circt_lec_args, "--assume-known-inputs"
        )
        has_explicit_canonicalizer_budget = (
            has_explicit_canonicalizer_max_iterations
            or has_explicit_canonicalizer_max_num_rewrites
        )
        verilog_timescale_override: str | None = None
        if verilog_timescale_fallback_mode == "on" and not has_explicit_timescale:
            verilog_timescale_override = verilog_fallback_timescale
        verilog_allow_multi_always_comb_drivers = (
            verilog_always_comb_multi_driver_mode == "on" and not has_explicit_multi_driver
        )
        verilog_synthesis_define_mode = (
            os.environ.get("LEC_VERILOG_SYNTHESIS_DEFINE_MODE", "auto").strip().lower()
        )
        if verilog_synthesis_define_mode not in {"auto", "on", "off"}:
            fail(
                "invalid LEC_VERILOG_SYNTHESIS_DEFINE_MODE: "
                f"{verilog_synthesis_define_mode} (expected auto|on|off)"
            )
        verilog_define_synthesis = False
        if verilog_synthesis_define_mode == "on":
            verilog_define_synthesis = True
        elif verilog_synthesis_define_mode == "auto":
            verilog_define_synthesis = not has_preprocessor_define(
                circt_verilog_args, "SYNTHESIS"
            )
        # Learned per-run frontend retry state. This carries successful fallback
        # settings across later batches, preventing repeated expensive
        # rediscovery of the same retry knobs on OpenTitan rule groups.
        learned_verilog_timescale_override = verilog_timescale_override
        learned_verilog_allow_multi_driver = verilog_allow_multi_always_comb_drivers
        learned_verilog_max_rss_mb: int | None = None
        # Learned per-run LEC retry state. Once a canonicalizer-timeout retry
        # proved necessary, start later cases with the bounded budget enabled
        # to avoid repeated timeout-first retries.
        #
        # In auto mode for low-timeout Z3 runs, pre-enable the bounded budget
        # from case 1 to avoid deterministic first-case timeout churn on known
        # OpenTitan timeout-frontier rules.
        learned_disable_threading = False
        auto_preenable_canonicalizer_timeout_budget = (
            canonicalizer_timeout_retry_mode == "auto"
            and lec_run_smtlib
            and not lec_smoke_only
            and not has_explicit_canonicalizer_budget
            and timeout_secs > 0
            and canonicalizer_timeout_retry_auto_preenable_timeout_secs > 0
            and timeout_secs
            <= canonicalizer_timeout_retry_auto_preenable_timeout_secs
        )
        learned_canonicalizer_timeout_budget = (
            auto_preenable_canonicalizer_timeout_budget
        )
        learned_canonicalizer_timeout_rewrite_budget: int | None = None
        if (
            learned_canonicalizer_timeout_budget
            and not has_explicit_canonicalizer_max_num_rewrites
            and canonicalizer_timeout_retry_max_num_rewrites > 0
        ):
            learned_canonicalizer_timeout_rewrite_budget = (
                canonicalizer_timeout_retry_max_num_rewrites
            )
        if auto_preenable_canonicalizer_timeout_budget:
            print(
                "opentitan connectivity lec: pre-enabling bounded canonicalizer "
                "budget for low-timeout z3 run "
                f"(timeout={timeout_secs}s<=auto-threshold="
                f"{canonicalizer_timeout_retry_auto_preenable_timeout_secs}s)",
                file=sys.stderr,
                flush=True,
            )
        # Learned per-run LEC retry state for LLHD abstraction UNKNOWN outcomes.
        learned_assume_known_inputs = has_assume_known_inputs

        case_batches: list[list[ConnectivityLECCase]] = []
        by_csv: dict[str, list[ConnectivityLECCase]] = {}
        for case in cases:
            csv_key = case.case_path.rsplit(":", 1)[0]
            if csv_key not in by_csv:
                by_csv[csv_key] = []
                case_batches.append(by_csv[csv_key])
            by_csv[csv_key].append(case)

        def append_timeout_reason(case: ConnectivityLECCase, reason: str) -> None:
            reason_key = (case.case_id, reason)
            if reason_key in timeout_reason_seen:
                return
            timeout_reason_seen.add(reason_key)
            timeout_reason_rows.append((case.case_id, case.case_path, reason))

        def append_drop_reasons(case: ConnectivityLECCase, reasons: list[str]) -> None:
            if reasons and case.case_id not in drop_remark_seen_cases:
                drop_remark_seen_cases.add(case.case_id)
                drop_remark_case_rows.append((case.case_id, case.case_path))
            for reason in reasons:
                key = (case.case_id, reason)
                if key in drop_remark_seen_case_reasons:
                    continue
                drop_remark_seen_case_reasons.add(key)
                drop_remark_reason_rows.append((case.case_id, case.case_path, reason))

        def split_batch_by_bind_top(
            batch_cases: list[ConnectivityLECCase],
        ) -> list[list[ConnectivityLECCase]]:
            by_top: dict[str, list[ConnectivityLECCase]] = {}
            ordered_tops: list[str] = []
            for case in batch_cases:
                top = case.bind_top
                if top not in by_top:
                    by_top[top] = []
                    ordered_tops.append(top)
                by_top[top].append(case)
            return [by_top[top] for top in ordered_tops]

        batch_counter = 0
        for initial_batch in case_batches:
            pending_batches: list[list[ConnectivityLECCase]] = [initial_batch]
            while pending_batches:
                batch_cases = pending_batches.pop(0)
                batch_tops = {case.bind_top for case in batch_cases}
                if len(batch_tops) > 1:
                    split_batches = split_batch_by_bind_top(batch_cases)
                    for split_batch in reversed(split_batches):
                        pending_batches.insert(0, split_batch)
                    continue
                batch_top = next(iter(batch_tops))
                batch_source_files = source_files
                if (
                    batch_top != default_top_module
                    and batch_top not in fallback_tops_used
                ):
                    batch_source_files = apply_top_override_source_prune(
                        batch_source_files, default_top_module, batch_top
                    )
                for fallback_top in sorted(fallback_tops_used):
                    if batch_top == fallback_top:
                        continue
                    batch_source_files = apply_top_override_source_prune(
                        batch_source_files, fallback_top, batch_top
                    )
                if not batch_source_files:
                    fail(
                        "no source files resolved after per-rule top override pruning: "
                        f"eda_top={default_top_module} case_top={batch_top}"
                    )
                batch_index = batch_counter
                batch_counter += 1

                shared_dir = workdir / "shared" / f"batch_{batch_index}"
                shared_dir.mkdir(parents=True, exist_ok=True)
                shared_verilog_log = shared_dir / "circt-verilog.log"
                shared_opt_log = shared_dir / "circt-opt.log"
                shared_moore_mlir = shared_dir / "connectivity.moore.mlir"
                shared_core_mlir = shared_dir / "connectivity.core.mlir"
                shared_core_mlirbc = shared_dir / "connectivity.core.mlirbc"
                shared_checker_sv = checks_dir / f"__circt_connectivity_batch_{batch_index}.sv"
                shared_missing_timescale_log = (
                    shared_dir / "circt-verilog.missing-timescale.log"
                )
                shared_always_comb_log = (
                    shared_dir / "circt-verilog.always-comb-multi-driver.log"
                )
                shared_resource_guard_log = (
                    shared_dir / "circt-verilog.resource-guard-rss.log"
                )
                shared_opt_emit_bytecode_log = (
                    shared_dir / "circt-opt.emit-bytecode.log"
                )

                with shared_checker_sv.open("w", encoding="utf-8") as handle:
                    for case in batch_cases:
                        text = case.checker_sv.read_text(encoding="utf-8")
                        handle.write(text)
                        if text and not text.endswith("\n"):
                            handle.write("\n")

                batch_timescale_override = learned_verilog_timescale_override
                batch_allow_multi_driver = learned_verilog_allow_multi_driver
                batch_verilog_max_rss_mb = learned_verilog_max_rss_mb
                batch_opt_emit_bytecode = opt_emit_bytecode_mode != "off"
                shared_core_input = (
                    shared_core_mlirbc if batch_opt_emit_bytecode else shared_core_mlir
                )

                def build_shared_verilog_cmd(
                    timescale_override: str | None,
                    allow_multi_driver: bool,
                    max_rss_mb: int | None,
                ) -> list[str]:
                    cmd = [
                        circt_verilog,
                        "--ir-moore",
                        "-o",
                        str(shared_moore_mlir),
                        "--single-unit",
                        "--no-uvm-auto-include",
                    ]
                    for case in batch_cases:
                        cmd.append(f"--top={case.ref_module}")
                        cmd.append(f"--top={case.impl_module}")
                    for include_dir in include_dirs:
                        cmd += ["-I", include_dir]
                    if timescale_override is not None and not has_explicit_timescale:
                        cmd.append(f"--timescale={timescale_override}")
                    if allow_multi_driver and not has_explicit_multi_driver:
                        cmd.append("--allow-multi-always-comb-drivers")
                    if max_rss_mb is not None and not has_explicit_verilog_resource_guard_policy:
                        cmd.append(f"--max-rss-mb={max_rss_mb}")
                    if verilog_define_synthesis:
                        cmd.append("-DSYNTHESIS")
                    cmd += circt_verilog_args
                    cmd += batch_source_files + [str(shared_checker_sv)]
                    return cmd

                shared_verilog_cmd = build_shared_verilog_cmd(
                    batch_timescale_override,
                    batch_allow_multi_driver,
                    batch_verilog_max_rss_mb,
                )
                def build_shared_opt_cmd(emit_bytecode: bool) -> list[str]:
                    output_path = shared_core_mlirbc if emit_bytecode else shared_core_mlir
                    cmd = [
                        circt_opt,
                        str(shared_moore_mlir),
                        "--moore-lower-concatref",
                        "--convert-moore-to-core",
                        "--mlir-disable-threading",
                    ]
                    if emit_bytecode:
                        cmd.append("--emit-bytecode")
                    cmd += ["-o", str(output_path)]
                    cmd += circt_opt_args
                    return cmd

                shared_opt_cmd = build_shared_opt_cmd(batch_opt_emit_bytecode)

                def mirror_shared_frontend_logs(case_dir: Path) -> None:
                    if shared_verilog_log.exists():
                        shutil.copy2(shared_verilog_log, case_dir / "circt-verilog.log")
                    if shared_opt_log.exists():
                        shutil.copy2(shared_opt_log, case_dir / "circt-opt.log")
                    if shared_opt_emit_bytecode_log.exists():
                        shutil.copy2(
                            shared_opt_emit_bytecode_log,
                            case_dir / "circt-opt.emit-bytecode.log",
                        )
                    if shared_missing_timescale_log.exists():
                        shutil.copy2(
                            shared_missing_timescale_log,
                            case_dir / "circt-verilog.missing-timescale.log",
                        )
                    if shared_always_comb_log.exists():
                        shutil.copy2(
                            shared_always_comb_log,
                            case_dir / "circt-verilog.always-comb-multi-driver.log",
                        )
                    if shared_resource_guard_log.exists():
                        shutil.copy2(
                            shared_resource_guard_log,
                            case_dir / "circt-verilog.resource-guard-rss.log",
                        )

                frontend_ok = False
                frontend_split = False
                frontend_force_singleton_split = False
                stage = "verilog"
                try:
                    attempted_timescale_retry = False
                    attempted_multi_driver_retry = False
                    while True:
                        try:
                            run_and_log(shared_verilog_cmd, shared_verilog_log, timeout_secs)
                            break
                        except subprocess.CalledProcessError:
                            if shared_verilog_log.is_file():
                                shared_verilog_log_text = shared_verilog_log.read_text(
                                    encoding="utf-8"
                                )
                            else:
                                shared_verilog_log_text = ""
                            if (
                                verilog_timescale_fallback_mode == "auto"
                                and batch_timescale_override is None
                                and not has_explicit_timescale
                                and not attempted_timescale_retry
                                and is_missing_timescale_retryable_failure(
                                    shared_verilog_log_text
                                )
                            ):
                                shutil.copy2(shared_verilog_log, shared_missing_timescale_log)
                                batch_timescale_override = verilog_fallback_timescale
                                learned_verilog_timescale_override = (
                                    batch_timescale_override
                                )
                                attempted_timescale_retry = True
                                print(
                                    "opentitan connectivity lec: retrying circt-verilog with "
                                    f"--timescale={batch_timescale_override} for "
                                    f"batch={batch_index}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                shared_verilog_cmd = build_shared_verilog_cmd(
                                    batch_timescale_override,
                                    batch_allow_multi_driver,
                                    batch_verilog_max_rss_mb,
                                )
                                continue
                            if (
                                verilog_always_comb_multi_driver_mode == "auto"
                                and not batch_allow_multi_driver
                                and not has_explicit_multi_driver
                                and not attempted_multi_driver_retry
                                and is_always_comb_multi_driver_retryable_failure(
                                    shared_verilog_log_text
                                )
                            ):
                                shutil.copy2(shared_verilog_log, shared_always_comb_log)
                                batch_allow_multi_driver = True
                                learned_verilog_allow_multi_driver = True
                                attempted_multi_driver_retry = True
                                print(
                                    "opentitan connectivity lec: retrying circt-verilog with "
                                    "--allow-multi-always-comb-drivers for "
                                    f"batch={batch_index}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                shared_verilog_cmd = build_shared_verilog_cmd(
                                    batch_timescale_override,
                                    batch_allow_multi_driver,
                                    batch_verilog_max_rss_mb,
                                )
                                continue
                            if (
                                verilog_auto_relax_resource_guard
                                and verilog_auto_relax_resource_guard_rss_ladder_mb
                                and not has_explicit_verilog_resource_guard_policy
                                and is_resource_guard_rss_retryable_failure(
                                    shared_verilog_log_text
                                )
                            ):
                                next_rss_mb: int | None = None
                                current_rss_mb = batch_verilog_max_rss_mb
                                for candidate in (
                                    verilog_auto_relax_resource_guard_rss_ladder_mb
                                ):
                                    if current_rss_mb is None or candidate > current_rss_mb:
                                        next_rss_mb = candidate
                                        break
                                if next_rss_mb is not None:
                                    shutil.copy2(
                                        shared_verilog_log, shared_resource_guard_log
                                    )
                                    batch_verilog_max_rss_mb = next_rss_mb
                                    learned_verilog_max_rss_mb = next_rss_mb
                                    print(
                                        "opentitan connectivity lec: retrying circt-verilog "
                                        f"with --max-rss-mb={next_rss_mb} for "
                                        f"batch={batch_index}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    shared_verilog_cmd = build_shared_verilog_cmd(
                                        batch_timescale_override,
                                        batch_allow_multi_driver,
                                        batch_verilog_max_rss_mb,
                                    )
                                    continue
                            if (
                                stage == "verilog"
                                and batch_allow_multi_driver
                                and is_always_comb_multi_driver_retryable_failure(
                                    shared_verilog_log_text
                                )
                            ):
                                frontend_force_singleton_split = True
                            raise

                    if shared_verilog_log.exists():
                        shared_reasons = extract_drop_reasons(
                            shared_verilog_log.read_text(encoding="utf-8"),
                            drop_remark_pattern,
                        )
                        for case in batch_cases:
                            append_drop_reasons(case, shared_reasons)
                    strip_vpi_attributes_for_opt(shared_moore_mlir)
                    stage = "opt"
                    attempted_opt_emit_bytecode_retry = False
                    while True:
                        try:
                            run_and_log(shared_opt_cmd, shared_opt_log, timeout_secs)
                            shared_core_input = (
                                shared_core_mlirbc
                                if batch_opt_emit_bytecode
                                else shared_core_mlir
                            )
                            break
                        except subprocess.CalledProcessError:
                            shared_opt_log_text = (
                                shared_opt_log.read_text(encoding="utf-8")
                                if shared_opt_log.is_file()
                                else ""
                            )
                            if (
                                opt_emit_bytecode_mode == "auto"
                                and batch_opt_emit_bytecode
                                and not attempted_opt_emit_bytecode_retry
                                and is_opt_emit_bytecode_retryable_failure(
                                    shared_opt_log_text
                                )
                            ):
                                shutil.copy2(
                                    shared_opt_log, shared_opt_emit_bytecode_log
                                )
                                batch_opt_emit_bytecode = False
                                attempted_opt_emit_bytecode_retry = True
                                print(
                                    "opentitan connectivity lec: retrying circt-opt "
                                    "without --emit-bytecode for "
                                    f"batch={batch_index}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                shared_opt_cmd = build_shared_opt_cmd(
                                    batch_opt_emit_bytecode
                                )
                                continue
                            raise
                    frontend_ok = True
                except subprocess.TimeoutExpired:
                    if len(batch_cases) > 1:
                        split_point = max(1, len(batch_cases) // 2)
                        pending_batches.insert(0, batch_cases[split_point:])
                        pending_batches.insert(0, batch_cases[:split_point])
                        frontend_split = True
                    else:
                        frontend_diag = (
                            "CIRCT_VERILOG_TIMEOUT"
                            if stage == "verilog"
                            else "CIRCT_OPT_TIMEOUT"
                        )
                        for case in batch_cases:
                            case_dir = workdir / "cases" / sanitize_token(case.case_id)
                            case_dir.mkdir(parents=True, exist_ok=True)
                            mirror_shared_frontend_logs(case_dir)
                            rows.append(
                                (
                                    "TIMEOUT",
                                    case.case_id,
                                    case.case_path,
                                    "opentitan",
                                    args.mode_label,
                                    frontend_diag,
                                )
                            )
                            append_timeout_reason(case, "frontend_command_timeout")
                except subprocess.CalledProcessError:
                    if len(batch_cases) > 1:
                        if frontend_force_singleton_split:
                            print(
                                "opentitan connectivity lec: splitting batch into "
                                "single-case frontends after persistent "
                                "always_comb multi-driver failure for "
                                f"batch={batch_index}",
                                file=sys.stderr,
                                flush=True,
                            )
                            for split_case in reversed(batch_cases):
                                pending_batches.insert(0, [split_case])
                        else:
                            split_point = max(1, len(batch_cases) // 2)
                            pending_batches.insert(0, batch_cases[split_point:])
                            pending_batches.insert(0, batch_cases[:split_point])
                        frontend_split = True
                    else:
                        frontend_diag = (
                            "CIRCT_VERILOG_ERROR"
                            if stage == "verilog"
                            else "CIRCT_OPT_ERROR"
                        )
                        for case in batch_cases:
                            case_dir = workdir / "cases" / sanitize_token(case.case_id)
                            case_dir.mkdir(parents=True, exist_ok=True)
                            mirror_shared_frontend_logs(case_dir)
                            rows.append(
                                (
                                    "ERROR",
                                    case.case_id,
                                    case.case_path,
                                    "opentitan",
                                    args.mode_label,
                                    frontend_diag,
                                )
                            )

                if frontend_split or not frontend_ok:
                    continue

                for case in batch_cases:
                    case_dir = workdir / "cases" / sanitize_token(case.case_id)
                    case_dir.mkdir(parents=True, exist_ok=True)
                    mirror_shared_frontend_logs(case_dir)
                    lec_log = case_dir / "circt-lec.log"
                    lec_out = case_dir / "circt-lec.out"
                    lec_enable_temporal_approx = (
                        temporal_approx_mode == "on" and not has_explicit_temporal_approx
                    )
                    lec_enable_disable_threading = learned_disable_threading
                    lec_enable_assume_known_inputs = learned_assume_known_inputs
                    lec_enable_canonicalizer_timeout_budget = (
                        learned_canonicalizer_timeout_budget
                    )
                    lec_canonicalizer_timeout_rewrite_budget = (
                        learned_canonicalizer_timeout_rewrite_budget
                    )
                    attempted_disable_threading_retry = False
                    attempted_llhd_abstraction_retry = False
                    attempted_canonicalizer_timeout_retry = False
                    can_retry_disable_threading = (
                        disable_threading_retry_mode == "on"
                        or (
                            disable_threading_retry_mode == "auto"
                            and not has_explicit_disable_threading
                        )
                    )
                    can_retry_canonicalizer_timeout = (
                        canonicalizer_timeout_retry_mode == "on"
                        or (
                            canonicalizer_timeout_retry_mode == "auto"
                            and not has_explicit_canonicalizer_budget
                        )
                    )
                    can_retry_llhd_abstraction = (
                        llhd_abstraction_assume_known_inputs_retry_mode == "on"
                        or (
                            llhd_abstraction_assume_known_inputs_retry_mode == "auto"
                            and not has_assume_known_inputs
                        )
                    )

                    def build_lec_cmd(
                        enable_temporal_approx: bool,
                        enable_disable_threading: bool,
                        enable_assume_known_inputs: bool,
                        enable_canonicalizer_timeout_budget: bool,
                        canonicalizer_timeout_rewrite_budget: int | None,
                    ) -> list[str]:
                        cmd = [
                            circt_lec,
                            str(shared_core_input),
                            f"-c1={case.ref_module}",
                            f"-c2={case.impl_module}",
                        ]
                        if lec_smoke_only:
                            cmd.append("--emit-mlir")
                        elif lec_run_smtlib:
                            cmd.append("--run-smtlib")
                            cmd.append(f"--z3-path={z3_bin}")
                        cmd += circt_lec_args
                        if enable_temporal_approx and not has_explicit_temporal_approx:
                            cmd.append("--approx-temporal")
                        if (
                            enable_disable_threading
                            and not has_explicit_disable_threading
                        ):
                            cmd.append("--mlir-disable-threading")
                        if enable_assume_known_inputs and not has_assume_known_inputs:
                            cmd.append("--assume-known-inputs")
                        if (
                            enable_canonicalizer_timeout_budget
                            and not has_explicit_canonicalizer_max_iterations
                            and canonicalizer_timeout_retry_max_iterations > 0
                        ):
                            cmd.append(
                                "--lec-canonicalizer-max-iterations="
                                f"{canonicalizer_timeout_retry_max_iterations}"
                            )
                        if (
                            enable_canonicalizer_timeout_budget
                            and not has_explicit_canonicalizer_max_num_rewrites
                        ):
                            rewrite_budget = canonicalizer_timeout_retry_max_num_rewrites
                            if canonicalizer_timeout_rewrite_budget is not None:
                                rewrite_budget = canonicalizer_timeout_rewrite_budget
                            if rewrite_budget > 0:
                                cmd.append(
                                    "--lec-canonicalizer-max-num-rewrites="
                                    f"{rewrite_budget}"
                                )
                        if verify_each_mode in {"auto", "off"} and not has_explicit_verify_each:
                            cmd.append("--verify-each=false")
                        return cmd

                    def next_lower_canonicalizer_rewrite_budget(
                        current_budget: int | None,
                    ) -> int | None:
                        if current_budget is None or current_budget <= 0:
                            current = canonicalizer_timeout_retry_max_num_rewrites
                        else:
                            current = current_budget
                        if current <= 0:
                            return None
                        for candidate in canonicalizer_timeout_retry_rewrite_ladder:
                            if candidate < current:
                                return candidate
                        return None

                    lec_cmd: list[str] = []
                    try:
                        attempted_temporal_retry = False
                        while True:
                            lec_cmd = build_lec_cmd(
                                lec_enable_temporal_approx,
                                lec_enable_disable_threading,
                                lec_enable_assume_known_inputs,
                                lec_enable_canonicalizer_timeout_budget,
                                lec_canonicalizer_timeout_rewrite_budget,
                            )
                            lec_timeout_secs = timeout_secs
                            if (
                                lec_enable_canonicalizer_timeout_budget
                                and canonicalizer_timeout_retry_timeout_secs > 0
                            ):
                                lec_timeout_secs = canonicalizer_timeout_retry_timeout_secs
                            try:
                                combined = run_and_log(
                                    lec_cmd, lec_log, lec_timeout_secs, out_path=lec_out
                                )
                                run_result = parse_lec_result(combined)
                                run_diag = parse_lec_diag(combined)
                                if (
                                    not lec_smoke_only
                                    and can_retry_llhd_abstraction
                                    and not lec_enable_assume_known_inputs
                                    and not attempted_llhd_abstraction_retry
                                    and is_llhd_abstraction_unknown(
                                        run_result, run_diag
                                    )
                                ):
                                    abstraction_retry_log = (
                                        case_dir / "circt-lec.llhd-abstraction.log"
                                    )
                                    if lec_log.is_file():
                                        shutil.copy2(lec_log, abstraction_retry_log)
                                    else:
                                        abstraction_retry_log.write_text(
                                            combined, encoding="utf-8"
                                        )
                                    lec_enable_assume_known_inputs = True
                                    learned_assume_known_inputs = True
                                    attempted_llhd_abstraction_retry = True
                                    print(
                                        "opentitan connectivity lec: retrying circt-lec with "
                                        "--assume-known-inputs for "
                                        f"{case.case_id}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    continue
                                break
                            except subprocess.TimeoutExpired:
                                if (
                                    not lec_smoke_only
                                    and can_retry_canonicalizer_timeout
                                    and not lec_enable_canonicalizer_timeout_budget
                                    and not attempted_canonicalizer_timeout_retry
                                ):
                                    timeout_retry_log = (
                                        case_dir / "circt-lec.canonicalizer-timeout.log"
                                    )
                                    if lec_log.is_file():
                                        shutil.copy2(lec_log, timeout_retry_log)
                                    else:
                                        timeout_retry_log.write_text("", encoding="utf-8")
                                    lec_enable_canonicalizer_timeout_budget = True
                                    learned_canonicalizer_timeout_budget = True
                                    if (
                                        lec_canonicalizer_timeout_rewrite_budget is None
                                        and not has_explicit_canonicalizer_max_num_rewrites
                                        and canonicalizer_timeout_retry_max_num_rewrites
                                        > 0
                                    ):
                                        lec_canonicalizer_timeout_rewrite_budget = (
                                            canonicalizer_timeout_retry_max_num_rewrites
                                        )
                                    if (
                                        lec_canonicalizer_timeout_rewrite_budget is not None
                                    ):
                                        learned_canonicalizer_timeout_rewrite_budget = (
                                            lec_canonicalizer_timeout_rewrite_budget
                                        )
                                    attempted_canonicalizer_timeout_retry = True
                                    timeout_transition = f"{lec_timeout_secs}s"
                                    if canonicalizer_timeout_retry_timeout_secs > 0:
                                        timeout_transition += (
                                            "->"
                                            f"{canonicalizer_timeout_retry_timeout_secs}s"
                                        )
                                    print(
                                        "opentitan connectivity lec: retrying circt-lec with "
                                        "bounded canonicalizer budget for "
                                        f"{case.case_id} "
                                        f"(timeout={timeout_transition})",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    continue
                                if (
                                    not lec_smoke_only
                                    and can_retry_canonicalizer_timeout
                                    and lec_enable_canonicalizer_timeout_budget
                                    and not has_explicit_canonicalizer_max_num_rewrites
                                ):
                                    current_rewrite_budget = (
                                        lec_canonicalizer_timeout_rewrite_budget
                                    )
                                    if (
                                        current_rewrite_budget is None
                                        and canonicalizer_timeout_retry_max_num_rewrites
                                        > 0
                                    ):
                                        current_rewrite_budget = (
                                            canonicalizer_timeout_retry_max_num_rewrites
                                        )
                                    next_rewrite_budget = (
                                        next_lower_canonicalizer_rewrite_budget(
                                            current_rewrite_budget
                                        )
                                    )
                                    if next_rewrite_budget is not None:
                                        timeout_retry_log = (
                                            case_dir
                                            / "circt-lec.canonicalizer-timeout-rewrite.log"
                                        )
                                        if lec_log.is_file():
                                            shutil.copy2(lec_log, timeout_retry_log)
                                        else:
                                            timeout_retry_log.write_text(
                                                "", encoding="utf-8"
                                            )
                                        lec_canonicalizer_timeout_rewrite_budget = (
                                            next_rewrite_budget
                                        )
                                        if (
                                            learned_canonicalizer_timeout_rewrite_budget
                                            is None
                                            or next_rewrite_budget
                                            < learned_canonicalizer_timeout_rewrite_budget
                                        ):
                                            learned_canonicalizer_timeout_rewrite_budget = (
                                                next_rewrite_budget
                                            )
                                        timeout_transition = f"{lec_timeout_secs}s"
                                        if (
                                            canonicalizer_timeout_retry_timeout_secs > 0
                                        ):
                                            timeout_transition += (
                                                "->"
                                                f"{canonicalizer_timeout_retry_timeout_secs}s"
                                            )
                                        print(
                                            "opentitan connectivity lec: retrying "
                                            "circt-lec with tighter canonicalizer "
                                            "rewrite budget for "
                                            f"{case.case_id} "
                                            "(max-num-rewrites="
                                            f"{current_rewrite_budget}->"
                                            f"{next_rewrite_budget}, "
                                            f"timeout={timeout_transition})",
                                            file=sys.stderr,
                                            flush=True,
                                        )
                                        continue
                                raise
                            except subprocess.CalledProcessError as exc:
                                if lec_log.is_file():
                                    lec_log_text = lec_log.read_text(encoding="utf-8")
                                else:
                                    lec_log_text = ""
                                lec_retry_combined = lec_log_text
                                if lec_out.is_file():
                                    lec_retry_combined += "\n" + lec_out.read_text(
                                        encoding="utf-8"
                                    )
                                if (
                                    can_retry_disable_threading
                                    and not lec_enable_disable_threading
                                    and not attempted_disable_threading_retry
                                    and is_disable_threading_retryable_failure(
                                        exc.returncode, lec_retry_combined
                                    )
                                ):
                                    disable_threading_retry_log = (
                                        case_dir / "circt-lec.disable-threading.log"
                                    )
                                    if lec_log.is_file():
                                        shutil.copy2(lec_log, disable_threading_retry_log)
                                    else:
                                        disable_threading_retry_log.write_text(
                                            lec_retry_combined, encoding="utf-8"
                                        )
                                    lec_enable_disable_threading = True
                                    learned_disable_threading = True
                                    attempted_disable_threading_retry = True
                                    print(
                                        "opentitan connectivity lec: retrying circt-lec with "
                                        "--mlir-disable-threading for "
                                        f"{case.case_id}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    continue
                                retry_result = parse_lec_result(lec_retry_combined)
                                retry_diag = parse_lec_diag(lec_retry_combined)
                                if (
                                    not lec_smoke_only
                                    and can_retry_llhd_abstraction
                                    and not lec_enable_assume_known_inputs
                                    and not attempted_llhd_abstraction_retry
                                    and is_llhd_abstraction_unknown(
                                        retry_result, retry_diag
                                    )
                                ):
                                    abstraction_retry_log = (
                                        case_dir / "circt-lec.llhd-abstraction.log"
                                    )
                                    if lec_log.is_file():
                                        shutil.copy2(lec_log, abstraction_retry_log)
                                    else:
                                        abstraction_retry_log.write_text(
                                            lec_retry_combined, encoding="utf-8"
                                        )
                                    lec_enable_assume_known_inputs = True
                                    learned_assume_known_inputs = True
                                    attempted_llhd_abstraction_retry = True
                                    print(
                                        "opentitan connectivity lec: retrying circt-lec with "
                                        "--assume-known-inputs for "
                                        f"{case.case_id}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    continue
                                if (
                                    temporal_approx_mode == "auto"
                                    and not has_explicit_temporal_approx
                                    and not lec_enable_temporal_approx
                                    and not attempted_temporal_retry
                                    and is_temporal_approx_retryable_failure(lec_log_text)
                                ):
                                    temporal_approx_log = (
                                        case_dir / "circt-lec.temporal-approx.log"
                                    )
                                    if lec_log.is_file():
                                        shutil.copy2(lec_log, temporal_approx_log)
                                    else:
                                        temporal_approx_log.write_text(
                                            lec_log_text, encoding="utf-8"
                                        )
                                    lec_enable_temporal_approx = True
                                    attempted_temporal_retry = True
                                    print(
                                        "opentitan connectivity lec: retrying circt-lec with "
                                        "--approx-temporal for "
                                        f"{case.case_id}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    continue
                                raise
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
                        timeout_reason = "runner_timeout_unknown_stage"
                        diag = "CIRCT_LEC_TIMEOUT"
                        if lec_smoke_only:
                            timeout_reason = "frontend_command_timeout"
                        elif lec_timeout_frontier_probe:
                            probe_cmd = [
                                arg
                                for arg in lec_cmd
                                if arg != "--run-smtlib" and not arg.startswith("--z3-path=")
                            ]
                            if "--emit-mlir" not in probe_cmd:
                                probe_cmd.append("--emit-mlir")
                            try:
                                run_and_log(
                                    probe_cmd,
                                    case_dir / "circt-lec.timeout-frontier.log",
                                    timeout_secs,
                                    out_path=case_dir / "circt-lec.timeout-frontier.out",
                                )
                            except subprocess.TimeoutExpired:
                                timeout_reason = "frontend_command_timeout"
                            except subprocess.CalledProcessError:
                                timeout_reason = "timeout_frontier_probe_error"
                            else:
                                timeout_reason = "solver_command_timeout"
                        else:
                            timeout_reason = "solver_command_timeout"
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
                        append_timeout_reason(case, timeout_reason)
                    except subprocess.CalledProcessError:
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

        governance_rc = evaluate_status_governance(rows)

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
        if args.timeout_reasons_file:
            timeout_path = Path(args.timeout_reasons_file).resolve()
            timeout_path.parent.mkdir(parents=True, exist_ok=True)
            with timeout_path.open("w", encoding="utf-8") as handle:
                for row in sorted(timeout_reason_rows, key=lambda item: (item[0], item[2])):
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
        if governance_rc != 0:
            counts["total"] += 1
            counts["error"] += 1
        print(
            "opentitan connectivity LEC summary: "
            f"total={counts['total']} pass={counts['pass']} fail={counts['fail']} "
            f"xfail={counts['xfail']} xpass={counts['xpass']} "
            f"error={counts['error']} skip={counts['skip']}",
            file=sys.stderr,
            flush=True,
        )
        case_rc = 0 if counts["fail"] == 0 and counts["error"] == 0 else 1
        return max(case_rc, governance_rc)
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

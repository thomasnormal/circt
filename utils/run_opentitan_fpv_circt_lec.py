#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Produce OpenTitan FPV LEC objective evidence from compile contracts.

This runner consumes OpenTitan FPV compile contracts and BMC-produced objective
manifests (assertion/cover TSVs). It executes a native CIRCT LEC health check
per selected FPV case and emits assertion-oriented objective evidence.

Cover objective evidence is optional and disabled by default because current
case-level LEC health checks do not derive native reachability semantics.
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
    setup_status: str
    toplevels: tuple[str, ...]
    files: tuple[str, ...]
    include_dirs: tuple[str, ...]
    defines: tuple[str, ...]


@dataclass(frozen=True)
class ObjectiveRow:
    kind: str
    bmc_status: str
    case_id: str
    case_path: str
    objective_id: str
    objective_label: str


@dataclass(frozen=True)
class CaseStatus:
    status: str
    diag: str
    reason: str


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


def parse_semicolon_list(raw: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in raw.split(";") if token.strip())


def parse_toplevels(raw: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def parse_case_id(raw: str) -> tuple[str, str]:
    case_id = raw.strip()
    if "::" not in case_id:
        fail(f"invalid case_id (expected target::toplevel): {case_id}")
    target_name, toplevel = case_id.split("::", 1)
    target_name = target_name.strip()
    toplevel = toplevel.strip()
    if not target_name or not toplevel:
        fail(f"invalid case_id (expected target::toplevel): {case_id}")
    return target_name, toplevel


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
                setup_status=setup_status,
                toplevels=parse_toplevels((row.get("toplevel") or "").strip()),
                files=parse_semicolon_list((row.get("files") or "").strip()),
                include_dirs=parse_semicolon_list((row.get("include_dirs") or "").strip()),
                defines=parse_semicolon_list((row.get("defines") or "").strip()),
            )
        )
    return out


def read_objective_rows(path: Path, kind: str) -> list[ObjectiveRow]:
    if not path.is_file():
        fail(f"{kind} objective source file not found: {path}")
    out: list[ObjectiveRow] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split("\t")
        if len(parts) < 5:
            fail(
                f"malformed {kind} objective row {idx} in {path}: expected >=5 TSV "
                f"columns"
            )
        out.append(
            ObjectiveRow(
                kind=kind,
                bmc_status=parts[0].strip().upper(),
                case_id=parts[1].strip(),
                case_path=parts[2].strip(),
                objective_id=parts[3].strip(),
                objective_label=parts[4].strip(),
            )
        )
    return out


def parse_lec_result(text: str) -> str | None:
    match = re.search(r"LEC_RESULT=(EQ|NEQ|UNKNOWN)", text)
    if match:
        return match.group(1)
    if re.search(r"\bc1 == c2\b", text):
        return "EQ"
    if re.search(r"\bc1 != c2\b", text):
        return "NEQ"
    return None


def write_log(path: Path, stdout: str, stderr: str) -> None:
    payload = ""
    if stdout:
        payload += stdout
        if not payload.endswith("\n"):
            payload += "\n"
    if stderr:
        payload += stderr
    path.write_text(payload, encoding="utf-8")


def run_with_log(
    cmd: list[str],
    log_path: Path,
    timeout_secs: int,
    *,
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
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    write_log(log_path, stdout, stderr)
    if out_path is not None:
        out_path.write_text(stdout, encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=stdout, stderr=stderr
        )
    return stdout + ("\n" + stderr if stderr else "")


def project_assertion_status(case_status: str, bmc_status: str) -> str:
    if case_status == "PASS":
        return "PROVEN"
    if case_status == "FAIL":
        return "FAILING"
    if case_status in {"UNKNOWN", "TIMEOUT", "SKIP", "ERROR"}:
        return case_status
    return "ERROR"


def project_cover_status(case_status: str, bmc_status: str) -> str:
    if case_status == "PASS":
        return "UNKNOWN"
    if case_status == "FAIL":
        return "UNKNOWN"
    if case_status in {"UNKNOWN", "TIMEOUT", "SKIP", "ERROR"}:
        return case_status
    return "ERROR"


def case_status_to_solver_result(case_status: str) -> str:
    return {
        "PASS": "EQ",
        "FAIL": "NEQ",
        "UNKNOWN": "UNKNOWN",
        "TIMEOUT": "TIMEOUT",
        "SKIP": "SKIP",
        "ERROR": "ERROR",
    }.get(case_status, "ERROR")


def evaluate_case(
    *,
    contract: ContractRow,
    toplevel: str,
    case_dir: Path,
    timeout_secs: int,
    circt_verilog: str,
    circt_verilog_args: list[str],
    circt_opt: str,
    circt_opt_args: list[str],
    circt_lec: str,
    circt_lec_args: list[str],
    lec_run_smtlib: bool,
    z3_bin: str,
) -> CaseStatus:
    case_dir.mkdir(parents=True, exist_ok=True)
    if contract.setup_status == "error":
        return CaseStatus(status="SKIP", diag="LEC_NOT_RUN", reason="setup_error")
    if not contract.files:
        return CaseStatus(status="SKIP", diag="LEC_NOT_RUN", reason="no_files")

    hw_mlir = case_dir / "input.hw.mlir"
    opt_mlir = case_dir / "input.opt.mlir"
    lec_out = case_dir / "lec.out.txt"

    v_cmd = [circt_verilog, *circt_verilog_args, "--ir-hw"]
    for incdir in contract.include_dirs:
        v_cmd.extend(["-I", incdir])
    for define in contract.defines:
        v_cmd.extend(["-D", define])
    v_cmd.extend(contract.files)
    try:
        run_with_log(
            v_cmd,
            case_dir / "circt-verilog.log",
            timeout_secs,
            out_path=hw_mlir,
        )
    except subprocess.TimeoutExpired:
        return CaseStatus(
            status="TIMEOUT", diag="CIRCT_VERILOG_TIMEOUT", reason="circt_verilog_timeout"
        )
    except subprocess.CalledProcessError:
        return CaseStatus(
            status="ERROR", diag="CIRCT_VERILOG_ERROR", reason="circt_verilog_error"
        )

    o_cmd = [circt_opt, *circt_opt_args, str(hw_mlir), "-o", str(opt_mlir)]
    try:
        run_with_log(o_cmd, case_dir / "circt-opt.log", timeout_secs)
    except subprocess.TimeoutExpired:
        return CaseStatus(
            status="TIMEOUT", diag="CIRCT_OPT_TIMEOUT", reason="circt_opt_timeout"
        )
    except subprocess.CalledProcessError:
        return CaseStatus(status="ERROR", diag="CIRCT_OPT_ERROR", reason="circt_opt_error")

    l_cmd = [circt_lec, *circt_lec_args]
    if lec_run_smtlib:
        l_cmd.append("--run-smtlib")
        if z3_bin:
            l_cmd.extend(["--z3-path", z3_bin])
    else:
        l_cmd.append("--run")
    l_cmd.extend(["--c1", toplevel, "--c2", toplevel, str(opt_mlir)])
    try:
        lec_text = run_with_log(
            l_cmd, case_dir / "circt-lec.log", timeout_secs, out_path=lec_out
        )
    except subprocess.TimeoutExpired:
        return CaseStatus(
            status="TIMEOUT", diag="CIRCT_LEC_TIMEOUT", reason="circt_lec_timeout"
        )
    except subprocess.CalledProcessError as exc:
        lec_text = (exc.output or "") + "\n" + (exc.stderr or "")
        lec_result = parse_lec_result(lec_text)
        if lec_result == "NEQ":
            return CaseStatus(status="FAIL", diag="LEC_RESULT_NEQ", reason="neq")
        if lec_result == "UNKNOWN":
            return CaseStatus(status="UNKNOWN", diag="LEC_RESULT_UNKNOWN", reason="unknown")
        return CaseStatus(status="ERROR", diag="CIRCT_LEC_ERROR", reason="circt_lec_error")

    lec_result = parse_lec_result(lec_text)
    if lec_result == "EQ":
        return CaseStatus(status="PASS", diag="LEC_RESULT_EQ", reason="eq")
    if lec_result == "NEQ":
        return CaseStatus(status="FAIL", diag="LEC_RESULT_NEQ", reason="neq")
    if lec_result == "UNKNOWN":
        return CaseStatus(status="UNKNOWN", diag="LEC_RESULT_UNKNOWN", reason="unknown")
    return CaseStatus(status="ERROR", diag="CIRCT_LEC_ERROR", reason="missing_lec_result")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenTitan FPV LEC objective evidence production."
    )
    parser.add_argument("--compile-contracts", required=True)
    parser.add_argument("--bmc-assertion-results", required=True)
    parser.add_argument("--bmc-cover-results", default="")
    parser.add_argument("--target-filter", default="")
    parser.add_argument(
        "--max-targets",
        default=os.environ.get("LEC_MAX_TARGETS", "0"),
        help="Optional maximum selected target count (0 means unlimited).",
    )
    parser.add_argument(
        "--target-shard-count",
        default=os.environ.get("LEC_TARGET_SHARD_COUNT", "1"),
        help="Optional deterministic target shard count (default: 1).",
    )
    parser.add_argument(
        "--target-shard-index",
        default=os.environ.get("LEC_TARGET_SHARD_INDEX", "0"),
        help="Optional deterministic shard index in [0, target-shard-count).",
    )
    parser.add_argument(
        "--results-file",
        default=os.environ.get("OUT", ""),
        help="Optional per-case LEC results TSV path.",
    )
    parser.add_argument(
        "--assertion-results-file",
        default=os.environ.get("LEC_ASSERTION_RESULTS_OUT", ""),
        help="Output per-assertion FPV LEC evidence TSV path.",
    )
    parser.add_argument(
        "--cover-results-file",
        default=os.environ.get("LEC_COVER_RESULTS_OUT", ""),
        help="Optional output per-cover FPV LEC evidence TSV path.",
    )
    parser.add_argument(
        "--emit-cover-evidence",
        action="store_true",
        default=os.environ.get("LEC_FPV_EMIT_COVER_EVIDENCE", "0") == "1",
        help=(
            "Emit FPV LEC cover objective rows. Disabled by default because "
            "case-level LEC health checks cannot yet derive native cover "
            "reachability semantics."
        ),
    )
    parser.add_argument("--workdir", default="")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument(
        "--mode-label",
        default=os.environ.get("LEC_MODE_LABEL", "FPV_LEC"),
        help="Mode label used for case rows (default: FPV_LEC).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    compile_contracts = Path(args.compile_contracts).resolve()
    bmc_assertions_path = Path(args.bmc_assertion_results).resolve()
    bmc_cover_path = Path(args.bmc_cover_results).resolve() if args.bmc_cover_results else None
    if not args.assertion_results_file:
        fail("missing --assertion-results-file (or LEC_ASSERTION_RESULTS_OUT)")

    results_path = Path(args.results_file).resolve() if args.results_file else None
    assertion_results_path = Path(args.assertion_results_file).resolve()
    cover_results_path = Path(args.cover_results_file).resolve() if args.cover_results_file else None

    max_targets = parse_nonnegative_int(args.max_targets, "max-targets")
    target_shard_count = parse_nonnegative_int(args.target_shard_count, "target-shard-count")
    target_shard_index = parse_nonnegative_int(args.target_shard_index, "target-shard-index")
    if target_shard_count < 1:
        fail("invalid --target-shard-count: expected integer >= 1")
    if target_shard_index >= target_shard_count:
        fail("invalid --target-shard-index: expected value < --target-shard-count")

    target_filter_re: re.Pattern[str] | None = None
    if args.target_filter:
        try:
            target_filter_re = re.compile(args.target_filter)
        except re.error as exc:
            fail(f"invalid --target-filter: {args.target_filter}: {exc}")

    contracts = read_compile_contracts(compile_contracts)
    if target_filter_re is not None:
        contracts = [
            row for row in contracts if target_filter_re.search(row.target_name) is not None
        ]

    ordered_targets: list[str] = []
    seen_targets: set[str] = set()
    for row in contracts:
        if row.target_name in seen_targets:
            continue
        seen_targets.add(row.target_name)
        ordered_targets.append(row.target_name)
    if max_targets > 0:
        ordered_targets = ordered_targets[:max_targets]
    selected_targets = {
        name
        for index, name in enumerate(ordered_targets)
        if (index % target_shard_count) == target_shard_index
    }
    contracts = [row for row in contracts if row.target_name in selected_targets]

    assertion_objectives = read_objective_rows(bmc_assertions_path, "assertion")
    cover_objectives: list[ObjectiveRow] = []
    if args.emit_cover_evidence and bmc_cover_path is not None and bmc_cover_path.is_file():
        cover_objectives = read_objective_rows(bmc_cover_path, "cover")

    objectives_by_case: dict[str, list[ObjectiveRow]] = {}
    for objective in assertion_objectives + cover_objectives:
        target_name, _ = parse_case_id(objective.case_id)
        if target_name not in selected_targets:
            continue
        objectives_by_case.setdefault(objective.case_id, []).append(objective)

    contract_case_map: dict[tuple[str, str], ContractRow] = {}
    for row in contracts:
        for toplevel in row.toplevels:
            contract_case_map[(row.target_name, toplevel)] = row

    case_ids = sorted(objectives_by_case.keys())

    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        cleanup_workdir = False
    else:
        workdir = Path(tempfile.mkdtemp(prefix="opentitan-fpv-lec-"))
        cleanup_workdir = not args.keep_workdir

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text("", encoding="utf-8")
    assertion_results_path.parent.mkdir(parents=True, exist_ok=True)
    assertion_results_path.write_text("", encoding="utf-8")
    if cover_results_path is not None:
        cover_results_path.parent.mkdir(parents=True, exist_ok=True)
        cover_results_path.write_text("", encoding="utf-8")

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
    z3_bin = os.environ.get("Z3_BIN", "")
    if lec_run_smtlib and not lec_smoke_only:
        if not z3_bin:
            z3_bin = shutil.which("z3") or ""
        if not z3_bin and Path.home().joinpath("z3-install/bin/z3").is_file():
            z3_bin = str(Path.home() / "z3-install/bin/z3")
        if not z3_bin and Path.home().joinpath("z3/build/z3").is_file():
            z3_bin = str(Path.home() / "z3/build/z3")
        if not z3_bin:
            fail("z3 not found; set Z3_BIN or disable LEC_RUN_SMTLIB")

    case_rows: list[tuple[str, str, str, str, str, str, str]] = []
    out_assertion_rows: list[tuple[str, str, str, str, str, str, str]] = []
    out_cover_rows: list[tuple[str, str, str, str, str, str, str]] = []

    counts = {
        "total": 0,
        "pass": 0,
        "fail": 0,
        "error": 0,
        "skip": 0,
        "unknown": 0,
        "timeout": 0,
    }

    for case_id in case_ids:
        target_name, toplevel = parse_case_id(case_id)
        case_key = (target_name, toplevel)
        case_objectives = objectives_by_case.get(case_id, [])
        case_path = case_objectives[0].case_path if case_objectives else ""
        contract = contract_case_map.get(case_key)
        if contract is None:
            case_status = CaseStatus(
                status="ERROR", diag="LEC_NOT_RUN", reason="missing_compile_contract_case"
            )
        else:
            case_status = evaluate_case(
                contract=contract,
                toplevel=toplevel,
                case_dir=workdir / target_name / toplevel,
                timeout_secs=timeout_secs,
                circt_verilog=circt_verilog,
                circt_verilog_args=circt_verilog_args,
                circt_opt=circt_opt,
                circt_opt_args=circt_opt_args,
                circt_lec=circt_lec,
                circt_lec_args=circt_lec_args,
                lec_run_smtlib=lec_run_smtlib and not lec_smoke_only,
                z3_bin=z3_bin,
            )
        counts["total"] += 1
        status_bucket = case_status.status.lower()
        if status_bucket in counts:
            counts[status_bucket] += 1
        else:
            counts["error"] += 1
        case_rows.append(
            (
                case_status.status,
                case_id,
                case_path,
                "opentitan",
                args.mode_label,
                case_status.diag,
                case_status.reason,
            )
        )
        solver_result = case_status_to_solver_result(case_status.status)
        for objective in case_objectives:
            if objective.kind == "assertion":
                projected = project_assertion_status(case_status.status, objective.bmc_status)
                out_assertion_rows.append(
                    (
                        projected,
                        objective.case_id,
                        objective.case_path,
                        objective.objective_id,
                        objective.objective_label,
                        solver_result,
                        case_status.reason,
                    )
                )
            else:
                projected = project_cover_status(case_status.status, objective.bmc_status)
                out_cover_rows.append(
                    (
                        projected,
                        objective.case_id,
                        objective.case_path,
                        objective.objective_id,
                        objective.objective_label,
                        solver_result,
                        case_status.reason,
                    )
                )

    if results_path is not None:
        with results_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            for row in case_rows:
                writer.writerow(row)
    with assertion_results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for row in out_assertion_rows:
            writer.writerow(row)
    if cover_results_path is not None:
        with cover_results_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            for row in out_cover_rows:
                writer.writerow(row)

    print(
        "opentitan fpv lec summary: "
        f"total={counts['total']} pass={counts['pass']} fail={counts['fail']} "
        f"error={counts['error']} skip={counts['skip']} unknown={counts['unknown']} "
        f"timeout={counts['timeout']} assertion_rows={len(out_assertion_rows)} "
        f"cover_rows={len(out_cover_rows)}",
        flush=True,
    )

    if cleanup_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

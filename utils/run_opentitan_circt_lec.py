#!/usr/bin/env python3
# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Run OpenTitan AES S-Box LEC checks with circt-lec.

This uses the existing OpenTitan AES S-Box LEC fixture set and compares each
implementation against the LUT-based reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_FORMAL_LIB_DIR = _THIS_DIR / "formal" / "lib"
if _FORMAL_LIB_DIR.is_dir():
    sys.path.insert(0, str(_FORMAL_LIB_DIR))


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


def has_command_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def replace_text(src: Path, dst: Path, replacements: list[tuple[str, str]]) -> None:
    data = src.read_text()
    for old, new in replacements:
        data = data.replace(old, new)
    dst.write_text(data)


def write_log(path: Path, stdout: str, stderr: str) -> None:
    data = ""
    if stdout:
        data += stdout
        if not data.endswith("\n"):
            data += "\n"
    if stderr:
        data += stderr
    if not data:
        data = ""
    path.write_text(data)

def strip_vpi_attributes_for_opt(path: Path) -> bool:
    """Strip trailing vpi.* op attributes that circt-opt may fail to parse."""
    text = path.read_text()
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
    path.write_text(out)
    return True

def strip_proc_assertions_global_for_lec(path: Path) -> bool:
    """Drop/replace proc-assertion helper globals that circt-lec cannot lower."""
    text = path.read_text()
    had_trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    changed = False
    kept: list[str] = []
    proc_assert_addr_vars: set[str] = set()
    addressof_re = re.compile(
        r"^(\s*%[A-Za-z0-9._$-]+)\s*=\s*llvm\.mlir\.addressof\s+"
        r"@__circt_proc_assertions_enabled\s*:\s*!llvm\.ptr(?:<[^>]+>)?\s*$"
    )
    cast_re = re.compile(
        r"^(\s*%[A-Za-z0-9._$-]+)\s*=\s*builtin\.unrealized_conversion_cast\s+"
        r"(%[A-Za-z0-9._$-]+)\s*:\s*.+\s+to\s+.+\s*$"
    )
    bitcast_re = re.compile(
        r"^(\s*%[A-Za-z0-9._$-]+)\s*=\s*llvm\.bitcast\s+"
        r"(%[A-Za-z0-9._$-]+)\s*:\s*!llvm\.ptr(?:<[^>]+>)?\s+to\s+"
        r"!llvm\.ptr(?:<[^>]+>)?\s*$"
    )
    addrspacecast_re = re.compile(
        r"^(\s*%[A-Za-z0-9._$-]+)\s*=\s*llvm\.addrspacecast\s+"
        r"(%[A-Za-z0-9._$-]+)\s*:\s*!llvm\.ptr(?:<[^>]+>)?\s+to\s+"
        r"!llvm\.ptr(?:<[^>]+>)?\s*$"
    )
    gep_re = re.compile(
        r"^(\s*%[A-Za-z0-9._$-]+)\s*=\s*llvm\.(?:getelementptr|gep)"
        r"(?:\s+inbounds)?\s+(%[A-Za-z0-9._$-]+)\s*\[[^\]]*\]\s*:\s*"
        r"\([^)]+\)\s*->\s*!llvm\.ptr(?:<[^>]+>)?\s*$"
    )
    load_re = re.compile(
        r"^(\s*%[A-Za-z0-9._$-]+)\s*=\s*llvm\.load(?:\s+volatile)?\s+"
        r"(%[A-Za-z0-9._$-]+)\s*(?:\{[^}]*\}\s*)?:\s*"
        r"!llvm\.ptr(?:<[^>]+>)?\s*->\s*i1(?:\s+\{[^}]*\})?\s*$"
    )
    for line in lines:
        if "llvm.mlir.global" in line and "__circt_proc_assertions_enabled" in line:
            changed = True
            continue
        m_addressof = addressof_re.match(line)
        if m_addressof:
            proc_assert_addr_vars.add(m_addressof.group(1).strip())
            changed = True
            continue
        m_cast = cast_re.match(line)
        if m_cast and m_cast.group(2).strip() in proc_assert_addr_vars:
            proc_assert_addr_vars.add(m_cast.group(1).strip())
            changed = True
            continue
        m_bitcast = bitcast_re.match(line)
        if m_bitcast and m_bitcast.group(2).strip() in proc_assert_addr_vars:
            proc_assert_addr_vars.add(m_bitcast.group(1).strip())
            changed = True
            continue
        m_addrspacecast = addrspacecast_re.match(line)
        if (
            m_addrspacecast
            and m_addrspacecast.group(2).strip() in proc_assert_addr_vars
        ):
            proc_assert_addr_vars.add(m_addrspacecast.group(1).strip())
            changed = True
            continue
        m_gep = gep_re.match(line)
        if m_gep and m_gep.group(2).strip() in proc_assert_addr_vars:
            proc_assert_addr_vars.add(m_gep.group(1).strip())
            changed = True
            continue
        m_load = load_re.match(line)
        if m_load and m_load.group(2) in proc_assert_addr_vars:
            kept.append(f"{m_load.group(1)} = hw.constant true")
            changed = True
            continue
        kept.append(line)
    if not changed:
        return False
    out = "\n".join(kept)
    if had_trailing_newline:
        out += "\n"
    path.write_text(out)
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

def parse_lec_diag_assume_known_result(text: str) -> str | None:
    match = re.search(r"LEC_DIAG_ASSUME_KNOWN_RESULT=(UNSAT|SAT|UNKNOWN)", text)
    if not match:
        return None
    return match.group(1)


def parse_xprop_summary_counts(text: str) -> dict[str, int]:
    # circt-lec prints a single diagnostics line in this form:
    #   summary: key1=V1 key2=V2 ...
    matches = re.findall(r"(?m)^.*summary:\s*(.*)$", text)
    if not matches:
        return {}
    counts: dict[str, int] = {}
    for key, value in re.findall(r"\b([a-z0-9][a-z0-9_-]*)=([0-9]+)\b", matches[-1]):
        counts[key] = int(value)
    return counts


def encode_summary_counts(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    return ",".join(f"{key}={counts[key]}" for key in sorted(counts))


def compute_contract_fingerprint(fields: list[str]) -> str:
    payload = "\x1f".join(fields).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


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
    out_path: Path | None = None,
    timeout_secs: int = 0,
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
            out_path.write_text(stdout)
        raise
    write_log(log_path, result.stdout, result.stderr)
    if out_path is not None:
        out_path.write_text(result.stdout)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    return result.stdout


try:
    from runner_common import (
        extract_drop_reasons as _shared_extract_drop_reasons,
        normalize_drop_reason as _shared_normalize_drop_reason,
        parse_nonnegative_int as _shared_parse_nonnegative_int,
        run_command_logged_with_env_retry as _shared_run_command_logged_with_env_retry,
        write_log as _shared_write_log,
    )
except Exception:
    _HAS_SHARED_FORMAL_HELPERS = False
else:
    _HAS_SHARED_FORMAL_HELPERS = True

try:
    from formal_results import write_results_jsonl_from_case_rows as _write_formal_results_jsonl_from_case_rows
    from formal_results import write_results_tsv as _write_formal_results_tsv
except Exception:
    _HAS_FORMAL_RESULT_SCHEMA = False
else:
    _HAS_FORMAL_RESULT_SCHEMA = True

if not _HAS_FORMAL_RESULT_SCHEMA:

    def _infer_stage(status: str, reason_code: str) -> str:
        status_norm = status.strip().upper()
        reason_norm = reason_code.strip().upper()
        if status_norm == "TIMEOUT":
            if "FRONTEND" in reason_norm:
                return "frontend"
            return "solver"
        if status_norm in {"ERROR", "FAIL"}:
            if "FRONTEND" in reason_norm:
                return "frontend"
            if "SMT" in reason_norm or "Z3" in reason_norm or "LEC" in reason_norm:
                return "solver"
        return "result"

    def _sort_formal_case_rows(
        rows: list[tuple[str, str, str, str, str, str]]
    ) -> list[tuple[str, str, str, str, str, str]]:
        return sorted(rows, key=lambda item: (item[1], item[0], item[2]))

    def _write_formal_results_tsv(
        path: Path, rows: list[tuple[str, str, str, str, str, str]]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in _sort_formal_case_rows(rows):
                handle.write("\t".join(row) + "\n")

    def _write_formal_results_jsonl_from_case_rows(
        path: Path,
        rows: list[tuple[str, str, str, str, str, str]],
        *,
        solver: str = "",
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for status, case_id, case_path, suite, mode, reason_code in _sort_formal_case_rows(rows):
                payload = {
                    "schema_version": 1,
                    "suite": suite,
                    "mode": mode,
                    "case_id": case_id,
                    "case_path": case_path,
                    "status": status.strip().upper(),
                    "reason_code": reason_code.strip().upper(),
                    "stage": _infer_stage(status, reason_code),
                    "solver": solver.strip(),
                    "solver_time_ms": None,
                    "frontend_time_ms": None,
                    "log_path": "",
                    "artifact_dir": "",
                }
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")

if _HAS_SHARED_FORMAL_HELPERS:

    def parse_nonnegative_int(raw: str, name: str) -> int:
        return _shared_parse_nonnegative_int(raw, name, fail)

    def normalize_drop_reason(line: str) -> str:
        return _shared_normalize_drop_reason(line)

    def extract_drop_reasons(log_text: str, pattern: str) -> list[str]:
        return _shared_extract_drop_reasons(log_text, pattern)

    def write_log(path: Path, stdout: str, stderr: str) -> None:
        _shared_write_log(path, stdout, stderr)

    def run_and_log(
        cmd: list[str],
        log_path: Path,
        out_path: Path | None = None,
        timeout_secs: int = 0,
    ) -> str:
        return _shared_run_command_logged_with_env_retry(
            cmd,
            log_path,
            timeout_secs=timeout_secs,
            out_path=out_path,
            fail_fn=fail,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OpenTitan AES S-Box LEC checks with circt-lec."
    )
    parser.add_argument(
        "--opentitan-root",
        default=str(Path.home() / "opentitan"),
        help="Path to OpenTitan checkout (default: ~/opentitan)",
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
        "--impl-filter",
        default="",
        help="Regex to filter S-Box implementations.",
    )
    parser.add_argument(
        "--include-masked",
        action="store_true",
        help="Include masked S-Box implementations.",
    )
    parser.add_argument(
        "--allow-invalid-op",
        action="store_true",
        help="Do not clamp invalid op_i values to a supported enum value.",
    )
    parser.add_argument(
        "--results-file",
        default=os.environ.get("OUT", ""),
        help="Optional TSV output path for per-implementation case rows.",
    )
    parser.add_argument(
        "--results-jsonl-file",
        default=os.environ.get("FORMAL_RESULTS_JSONL_OUT", ""),
        help="Optional JSONL output path for unified formal result schema rows.",
    )
    parser.add_argument(
        "--xprop-summary-file",
        default=os.environ.get("OUT_XPROP_SUMMARY", ""),
        help=(
            "Optional TSV output path for XPROP diagnostic rows "
            "(implementation, mode, diag, result, counters, log_dir, "
            "assume_known_result)."
        ),
    )
    parser.add_argument(
        "--drop-remark-cases-file",
        default=os.environ.get("LEC_DROP_REMARK_CASES_OUT", ""),
        help="Optional TSV output path for dropped-syntax case IDs.",
    )
    parser.add_argument(
        "--drop-remark-reasons-file",
        default=os.environ.get("LEC_DROP_REMARK_REASONS_OUT", ""),
        help="Optional TSV output path for dropped-syntax case+reason rows.",
    )
    parser.add_argument(
        "--timeout-reasons-file",
        default=os.environ.get("LEC_TIMEOUT_REASON_CASES_OUT", ""),
        help="Optional TSV output path for timeout reason rows.",
    )
    parser.add_argument(
        "--resolved-contracts-file",
        default=os.environ.get("LEC_RESOLVED_CONTRACTS_OUT", ""),
        help="Optional TSV output path for resolved per-case contract rows.",
    )
    args = parser.parse_args()

    ot_root = Path(args.opentitan_root).resolve()
    rtl_path = ot_root / "hw/ip/aes/rtl"
    prim_path = ot_root / "hw/ip/prim/rtl"
    prim_xilinx_path = ot_root / "hw/ip/prim_xilinx/rtl"
    wrapper_path = (
        ot_root / "hw/ip/aes/pre_dv/aes_sbox_lec/aes_sbox_masked_wrapper.sv"
    )

    if not rtl_path.is_dir():
        print(f"OpenTitan RTL path not found: {rtl_path}", file=sys.stderr)
        return 1

    impl_gold = "aes_sbox_lut"
    impl_list = [
        path.stem
        for path in rtl_path.glob("aes_sbox_*.sv")
        if path.suffix == ".sv"
    ]
    for item in ("aes_sbox_dom", impl_gold, "aes_sbox_canright_pkg"):
        if item in impl_list:
            impl_list.remove(item)
    if not args.include_masked:
        impl_list = [name for name in impl_list if "masked" not in name]
    if args.impl_filter:
        pattern = re.compile(args.impl_filter)
        impl_list = [name for name in impl_list if pattern.search(name)]

    if not impl_list:
        print("No AES S-Box implementations selected.", file=sys.stderr)
        return 1

    circt_verilog = os.environ.get("CIRCT_VERILOG", "build_test/bin/circt-verilog")
    circt_verilog_args = shlex.split(os.environ.get("CIRCT_VERILOG_ARGS", ""))
    circt_opt = os.environ.get("CIRCT_OPT", "build_test/bin/circt-opt")
    circt_opt_args = shlex.split(os.environ.get("CIRCT_OPT_ARGS", ""))
    circt_lec = os.environ.get("CIRCT_LEC", "build_test/bin/circt-lec")
    circt_lec_args = shlex.split(os.environ.get("CIRCT_LEC_ARGS", ""))
    has_explicit_verify_each = has_command_option(circt_lec_args, "--verify-each")
    # OpenTitan S-Box parity is evaluated with x-optimistic equivalence by
    # default to avoid classifying known-input-equivalent implementations as
    # strict XPROP-only failures.
    lec_x_optimistic = os.environ.get("LEC_X_OPTIMISTIC", "1") == "1"
    # OpenTitan strict-X runs default to known-input assumptions to avoid
    # spurious XPROP_ONLY deltas driven solely by unconstrained symbolic inputs.
    # Explicit LEC_ASSUME_KNOWN_INPUTS still takes precedence.
    lec_assume_known_inputs_env = os.environ.get("LEC_ASSUME_KNOWN_INPUTS", "")
    if lec_assume_known_inputs_env:
        lec_assume_known_inputs = lec_assume_known_inputs_env == "1"
    else:
        lec_assume_known_inputs = not lec_x_optimistic
    lec_diagnose_xprop = os.environ.get("LEC_DIAGNOSE_XPROP", "1") == "1"
    lec_dump_unknown_sources = (
        os.environ.get("LEC_DUMP_UNKNOWN_SOURCES", "0") == "1"
    )
    lec_accept_xprop_only = os.environ.get("LEC_ACCEPT_XPROP_ONLY", "0") == "1"
    # OpenTitan parity mode accepts known LLHD abstraction diagnostics by
    # default; set LEC_ACCEPT_LLHD_ABSTRACTION=0 to disable.
    lec_accept_llhd_abstraction = (
        os.environ.get("LEC_ACCEPT_LLHD_ABSTRACTION", "1") == "1"
    )
    lec_smoke_only = os.environ.get("LEC_SMOKE_ONLY", "0") == "1"
    lec_run_smtlib = os.environ.get("LEC_RUN_SMTLIB", "1") == "1"
    lec_timeout_frontier_probe = (
        os.environ.get("LEC_TIMEOUT_FRONTIER_PROBE", "1") == "1"
    )
    lec_mode_label = os.environ.get("LEC_MODE_LABEL", "LEC").strip() or "LEC"
    lec_verify_each_mode = os.environ.get("LEC_VERIFY_EACH_MODE", "auto").strip().lower()
    if lec_verify_each_mode not in {"auto", "on", "off"}:
        fail(
            "invalid LEC_VERIFY_EACH_MODE: "
            f"{lec_verify_each_mode} (expected auto|on|off)"
        )
    lec_timeout_secs = parse_nonnegative_int(
        os.environ.get("LEC_TIMEOUT_SECS", "0"),
        "LEC_TIMEOUT_SECS",
    )
    drop_remark_pattern = os.environ.get(
        "LEC_DROP_REMARK_PATTERN",
        os.environ.get("DROP_REMARK_PATTERN", "will be dropped during lowering"),
    )
    z3_bin = os.environ.get("Z3_BIN", "")
    if lec_run_smtlib and not lec_smoke_only:
        if not z3_bin:
            z3_bin = shutil.which("z3") or ""
        if not z3_bin and Path.home().joinpath("z3-install/bin/z3").is_file():
            z3_bin = str(Path.home() / "z3-install/bin/z3")
        if not z3_bin and Path.home().joinpath("z3/build/z3").is_file():
            z3_bin = str(Path.home() / "z3/build/z3")
        if not z3_bin:
            print("z3 not found; set Z3_BIN or disable LEC_RUN_SMTLIB",
                  file=sys.stderr)
            return 1
    clamp_invalid_op = not args.allow_invalid_op

    if lec_x_optimistic and "--x-optimistic" not in circt_lec_args:
        circt_lec_args.append("--x-optimistic")
    if lec_assume_known_inputs and "--assume-known-inputs" not in circt_lec_args:
        circt_lec_args.append("--assume-known-inputs")
    if lec_diagnose_xprop and "--diagnose-xprop" not in circt_lec_args:
        # Safe to pass even for non-solver modes; it is only acted on under
        # --run-smtlib.
        circt_lec_args.append("--diagnose-xprop")
    if (
        lec_dump_unknown_sources
        and not lec_smoke_only
        and lec_run_smtlib
        and "--dump-unknown-sources" not in circt_lec_args
    ):
        circt_lec_args.append("--dump-unknown-sources")
    if (
        lec_accept_llhd_abstraction
        and "--accept-llhd-abstraction" not in circt_lec_args
    ):
        circt_lec_args.append("--accept-llhd-abstraction")

    contract_source = "manifest"
    contract_backend_mode = "smoke"
    if not lec_smoke_only:
        contract_backend_mode = "smtlib" if lec_run_smtlib else "jit"
    contract_z3_path = z3_bin if contract_backend_mode == "smtlib" else ""
    contract_lec_args = shlex.join(circt_lec_args)

    def write_valid_op_wrapper(out_path: Path, wrapper_name: str, inner_name: str) -> None:
        out_path.write_text(
            "\n".join(
                [
                    "module " + wrapper_name + " (",
                    "  input  aes_pkg::ciph_op_e op_i,",
                    "  input  logic [7:0]        data_i,",
                    "  output logic [7:0]        data_o",
                    ");",
                    "  always_comb begin",
                    "    assume (op_i == aes_pkg::CIPH_FWD || op_i == aes_pkg::CIPH_INV);",
                    "  end",
                    "  " + inner_name + " u_dut (",
                    "    .op_i   ( op_i ),",
                    "    .data_i ( data_i ),",
                    "    .data_o ( data_o )",
                    "  );",
                    "endmodule",
                    "",
                ]
            )
        )

    def pick_prim(name: str, fallback: str) -> str:
        preferred = prim_xilinx_path / name
        if preferred.exists():
            return str(preferred)
        return str(prim_xilinx_path / fallback)

    dep_list = [
        str(rtl_path / "aes_pkg.sv"),
        str(rtl_path / "aes_reg_pkg.sv"),
        str(rtl_path / "aes_sbox_canright_pkg.sv"),
        str(prim_path / "prim_util_pkg.sv"),
        pick_prim("prim_xilinx_buf.sv", "prim_buf.sv"),
        pick_prim("prim_xilinx_xor2.sv", "prim_xor2.sv"),
    ]

    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        keep_workdir = True
    else:
        workdir = Path(tempfile.mkdtemp(prefix="opentitan-lec-"))
        keep_workdir = args.keep_workdir

    failures = 0
    case_rows: list[tuple[str, str, str, str, str, str]] = []
    xprop_rows: list[tuple[str, str, str, str, str, str, str, str]] = []
    drop_remark_case_rows: list[tuple[str, str]] = []
    drop_remark_reason_rows: list[tuple[str, str, str]] = []
    timeout_reason_rows: list[tuple[str, str, str]] = []
    resolved_contract_rows: list[tuple[str, ...]] = []
    drop_remark_seen_cases: set[str] = set()
    drop_remark_seen_case_reasons: set[tuple[str, str]] = set()
    timeout_reason_seen: set[tuple[str, str]] = set()
    try:
        print(
            f"Running LEC on {len(impl_list)} AES S-Box implementation(s)...",
            flush=True,
        )
        for impl in sorted(impl_list):
            impl_dir = workdir / impl
            impl_dir.mkdir(parents=True, exist_ok=True)
            verilog_log_path = impl_dir / "circt-verilog.log"
            contract_fields = [
                contract_source,
                contract_backend_mode,
                lec_mode_label,
                "1" if lec_x_optimistic else "0",
                "1" if lec_assume_known_inputs else "0",
                "1" if lec_accept_xprop_only else "0",
                "1" if lec_accept_llhd_abstraction else "0",
                "1" if lec_diagnose_xprop else "0",
                "1" if lec_dump_unknown_sources else "0",
                contract_z3_path,
                contract_lec_args,
            ]
            contract_fingerprint = compute_contract_fingerprint(contract_fields)
            resolved_contract_rows.append(
                (impl, str(impl_dir), *contract_fields, contract_fingerprint)
            )
            src_ref = rtl_path / f"{impl_gold}.sv"
            src_impl = rtl_path / f"{impl}.sv"
            ref_file = src_ref
            ref_module = impl_gold

            extra_files = []
            if "masked" in impl:
                wrapper_out = impl_dir / "aes_sbox_masked_wrapper.sv"
                replace_text(
                    wrapper_path,
                    wrapper_out,
                    [
                        ("aes_sbox_masked", impl),
                    ],
                )
                extra_files.append(str(wrapper_out))
                extra_files.append(str(src_impl))
                dut_module = f"{impl}_wrapper"
            else:
                extra_files.append(str(src_impl))
                dut_module = impl

            if clamp_invalid_op:
                ref_wrapper = f"{ref_module}_lec_wrapper"
                ref_wrapper_path = impl_dir / f"{ref_wrapper}.sv"
                write_valid_op_wrapper(ref_wrapper_path, ref_wrapper, ref_module)
                extra_files.append(str(ref_wrapper_path))
                ref_module = ref_wrapper

                dut_wrapper = f"{dut_module}_lec_wrapper"
                dut_wrapper_path = impl_dir / f"{dut_wrapper}.sv"
                write_valid_op_wrapper(dut_wrapper_path, dut_wrapper, dut_module)
                extra_files.append(str(dut_wrapper_path))
                dut_module = dut_wrapper

            out_moore = impl_dir / "aes_sbox_lec.moore.mlir"
            out_mlir = impl_dir / "aes_sbox_lec.mlir"
            verilog_cmd = [
                circt_verilog,
                "--ir-moore",
                "-o",
                str(out_moore),
                "--single-unit",
                "--no-uvm-auto-include",
                f"--top={ref_module}",
                f"--top={dut_module}",
                "-I",
                str(prim_path),
                "-I",
                str(prim_xilinx_path),
                "-I",
                str(rtl_path),
            ]
            verilog_cmd += circt_verilog_args
            verilog_cmd += dep_list + [str(ref_file)] + extra_files

            opt_cmd = [
                circt_opt,
                str(out_moore),
                "--convert-moore-to-core",
                "--mlir-disable-threading",
                "-o",
                str(out_mlir),
            ]
            opt_cmd += circt_opt_args

            lec_cmd = [
                circt_lec,
                str(out_mlir),
                f"-c1={ref_module}",
                f"-c2={dut_module}",
            ]
            if lec_smoke_only:
                lec_cmd.append("--emit-mlir")
            else:
                if lec_run_smtlib:
                    lec_cmd.append("--run-smtlib")
                    lec_cmd.append(f"--z3-path={z3_bin}")
            lec_cmd += circt_lec_args
            if lec_verify_each_mode in {"auto", "off"} and not has_explicit_verify_each:
                lec_cmd.append("--verify-each=false")

            result: str | None = None
            diag: str | None = None
            assume_known_result: str | None = None
            summary_counts: dict[str, int] = {}
            stage = "verilog"
            try:
                stage = "verilog"
                run_and_log(
                    verilog_cmd,
                    verilog_log_path,
                    timeout_secs=lec_timeout_secs,
                )
                strip_vpi_attributes_for_opt(out_moore)
                if verilog_log_path.exists():
                    reasons = extract_drop_reasons(
                        verilog_log_path.read_text(), drop_remark_pattern
                    )
                    if reasons:
                        if impl not in drop_remark_seen_cases:
                            drop_remark_seen_cases.add(impl)
                            drop_remark_case_rows.append((impl, str(impl_dir)))
                        for reason in reasons:
                            reason_key = (impl, reason)
                            if reason_key in drop_remark_seen_case_reasons:
                                continue
                            drop_remark_seen_case_reasons.add(reason_key)
                            drop_remark_reason_rows.append(
                                (impl, str(impl_dir), reason)
                            )
                stage = "opt"
                run_and_log(
                    opt_cmd,
                    impl_dir / "circt-opt.log",
                    timeout_secs=lec_timeout_secs,
                )
                strip_proc_assertions_global_for_lec(out_mlir)
                stage = "lec"
                lec_stdout = run_and_log(
                    lec_cmd,
                    impl_dir / "circt-lec.log",
                    out_path=impl_dir / "circt-lec.out",
                    timeout_secs=lec_timeout_secs,
                )
                if not lec_smoke_only:
                    lec_log_text = (impl_dir / "circt-lec.log").read_text()
                    combined = lec_log_text + "\n" + lec_stdout
                    result = parse_lec_result(combined)
                    diag = parse_lec_diag(combined)
                    assume_known_result = parse_lec_diag_assume_known_result(combined)
                    if diag == "XPROP_ONLY":
                        summary_counts = parse_xprop_summary_counts(combined)
                    # Fail closed: non-smoke LEC must provide an explicit
                    # equivalence result token, otherwise the run is
                    # non-actionable and must not be counted as PASS.
                    if result is None:
                        diag = "LEC_RESULT_MISSING"
                        raise subprocess.CalledProcessError(
                            1, lec_cmd, output=lec_stdout, stderr=lec_log_text
                        )
                    if result in ("NEQ", "UNKNOWN"):
                        if diag == "XPROP_ONLY" and lec_accept_xprop_only:
                            print(
                                f"{impl:24} XPROP_ONLY (accepted)",
                                flush=True,
                            )
                            case_rows.append(
                                (
                                    "XFAIL",
                                    impl,
                                    f"{impl_dir}#XPROP_ONLY",
                                    "opentitan",
                                    lec_mode_label,
                                    "XPROP_ONLY",
                                )
                            )
                            xprop_rows.append(
                                (
                                    "XFAIL",
                                    impl,
                                    lec_mode_label,
                                    "XPROP_ONLY",
                                    result or "",
                                    encode_summary_counts(summary_counts),
                                    str(impl_dir),
                                    assume_known_result or "",
                                )
                            )
                            continue
                        raise subprocess.CalledProcessError(
                            1, lec_cmd, output=lec_stdout, stderr=lec_log_text
                        )
                if not diag:
                    if result in ("EQ", "NEQ", "UNKNOWN"):
                        diag = result
                    elif lec_smoke_only:
                        diag = "SMOKE_ONLY"
                    else:
                        diag = "PASS"
                print(f"{impl:24} OK", flush=True)
                case_rows.append(
                    ("PASS", impl, str(impl_dir), "opentitan", lec_mode_label, diag)
                )
            except subprocess.TimeoutExpired:
                failures += 1
                timeout_reason = "runner_timeout_unknown_stage"
                if stage == "verilog":
                    diag = "CIRCT_VERILOG_TIMEOUT"
                    timeout_reason = "frontend_command_timeout"
                elif stage == "opt":
                    diag = "CIRCT_OPT_TIMEOUT"
                    timeout_reason = "frontend_command_timeout"
                elif stage == "lec":
                    diag = "CIRCT_LEC_TIMEOUT"
                    if lec_smoke_only:
                        timeout_reason = "frontend_command_timeout"
                    elif lec_timeout_frontier_probe:
                        probe_cmd = [
                            arg
                            for arg in lec_cmd
                            if arg != "--run-smtlib"
                            and not arg.startswith("--z3-path=")
                        ]
                        if "--emit-mlir" not in probe_cmd:
                            probe_cmd.append("--emit-mlir")
                        try:
                            run_and_log(
                                probe_cmd,
                                impl_dir / "circt-lec.timeout-frontier.log",
                                out_path=impl_dir / "circt-lec.timeout-frontier.out",
                                timeout_secs=lec_timeout_secs,
                            )
                        except subprocess.TimeoutExpired:
                            timeout_reason = "frontend_command_timeout"
                        except subprocess.CalledProcessError:
                            timeout_reason = "timeout_frontier_probe_error"
                        else:
                            timeout_reason = "solver_command_timeout"
                    else:
                        timeout_reason = "solver_command_timeout"
                else:
                    diag = "TIMEOUT"
                print(f"{impl:24} TIMEOUT ({diag}) (logs in {impl_dir})", flush=True)
                case_rows.append(
                    (
                        "TIMEOUT",
                        impl,
                        f"{impl_dir}#{diag}",
                        "opentitan",
                        lec_mode_label,
                        diag,
                    )
                )
                reason_key = (impl, timeout_reason)
                if reason_key not in timeout_reason_seen:
                    timeout_reason_seen.add(reason_key)
                    timeout_reason_rows.append((impl, str(impl_dir), timeout_reason))
            except subprocess.CalledProcessError:
                extra = ""
                try:
                    lec_log_text = (impl_dir / "circt-lec.log").read_text()
                    lec_out_text = (impl_dir / "circt-lec.out").read_text()
                    combined = lec_log_text + "\n" + lec_out_text
                    if result is None:
                        result = parse_lec_result(combined)
                    if diag is None:
                        diag = parse_lec_diag(combined)
                    if assume_known_result is None:
                        assume_known_result = parse_lec_diag_assume_known_result(combined)
                    if diag == "XPROP_ONLY":
                        summary_counts = parse_xprop_summary_counts(combined)
                    if diag:
                        extra = f" ({diag})"
                except Exception:
                    pass
                if stage == "lec" and not lec_smoke_only and result == "EQ":
                    if not diag:
                        diag = "EQ"
                    print(f"{impl:24} OK", flush=True)
                    case_rows.append(
                        (
                            "PASS",
                            impl,
                            str(impl_dir),
                            "opentitan",
                            lec_mode_label,
                            diag,
                        )
                    )
                    continue
                if (
                    stage == "lec"
                    and not lec_smoke_only
                    and result in ("NEQ", "UNKNOWN")
                    and diag == "XPROP_ONLY"
                    and lec_accept_xprop_only
                ):
                    print(
                        f"{impl:24} XPROP_ONLY (accepted)",
                        flush=True,
                    )
                    case_rows.append(
                        (
                            "XFAIL",
                            impl,
                            f"{impl_dir}#XPROP_ONLY",
                            "opentitan",
                            lec_mode_label,
                            "XPROP_ONLY",
                        )
                    )
                    xprop_rows.append(
                        (
                            "XFAIL",
                            impl,
                            lec_mode_label,
                            "XPROP_ONLY",
                            result or "",
                            encode_summary_counts(summary_counts),
                            str(impl_dir),
                            assume_known_result or "",
                        )
                    )
                    continue
                failures += 1
                if not diag:
                    if result in ("NEQ", "UNKNOWN", "EQ"):
                        diag = result
                    elif stage == "verilog":
                        diag = "CIRCT_VERILOG_ERROR"
                    elif stage == "opt":
                        diag = "CIRCT_OPT_ERROR"
                    elif stage == "lec" and lec_smoke_only:
                        diag = "SMOKE_ONLY_ERROR"
                    elif stage == "lec":
                        diag = "CIRCT_LEC_ERROR"
                    else:
                        diag = "ERROR"
                if diag and not extra:
                    extra = f" ({diag})"
                print(f"{impl:24} FAIL{extra} (logs in {impl_dir})", flush=True)
                detail = str(impl_dir)
                if diag:
                    detail = f"{detail}#{diag}"
                case_rows.append(
                    (
                        "FAIL",
                        impl,
                        detail,
                        "opentitan",
                        lec_mode_label,
                        diag or "",
                    )
                )
                if diag == "XPROP_ONLY":
                    xprop_rows.append(
                        (
                            "FAIL",
                            impl,
                            lec_mode_label,
                            "XPROP_ONLY",
                            result or "",
                            encode_summary_counts(summary_counts),
                            str(impl_dir),
                            assume_known_result or "",
                        )
                    )

    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    if args.results_file:
        results_path = Path(args.results_file)
        _write_formal_results_tsv(results_path, case_rows)
    if args.results_jsonl_file:
        results_jsonl_path = Path(args.results_jsonl_file)
        solver_label = "z3" if lec_run_smtlib and not lec_smoke_only else ""
        _write_formal_results_jsonl_from_case_rows(
            results_jsonl_path, case_rows, solver=solver_label
        )
    if args.xprop_summary_file:
        xprop_summary_path = Path(args.xprop_summary_file)
        xprop_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with xprop_summary_path.open("w") as handle:
            for row in sorted(xprop_rows, key=lambda item: (item[1], item[2], item[0], item[6])):
                handle.write("\t".join(row) + "\n")
    if args.drop_remark_cases_file:
        drop_case_path = Path(args.drop_remark_cases_file)
        drop_case_path.parent.mkdir(parents=True, exist_ok=True)
        with drop_case_path.open("w") as handle:
            for row in sorted(drop_remark_case_rows, key=lambda item: item[0]):
                handle.write("\t".join(row) + "\n")
    if args.drop_remark_reasons_file:
        drop_reason_path = Path(args.drop_remark_reasons_file)
        drop_reason_path.parent.mkdir(parents=True, exist_ok=True)
        with drop_reason_path.open("w") as handle:
            for row in sorted(drop_remark_reason_rows, key=lambda item: (item[0], item[2])):
                handle.write("\t".join(row) + "\n")
    if args.timeout_reasons_file:
        timeout_path = Path(args.timeout_reasons_file)
        timeout_path.parent.mkdir(parents=True, exist_ok=True)
        with timeout_path.open("w") as handle:
            for row in sorted(timeout_reason_rows, key=lambda item: (item[0], item[2])):
                handle.write("\t".join(row) + "\n")
    if args.resolved_contracts_file:
        contracts_path = Path(args.resolved_contracts_file)
        contracts_path.parent.mkdir(parents=True, exist_ok=True)
        with contracts_path.open("w") as handle:
            handle.write("#resolved_contract_schema_version=1\n")
            for row in sorted(resolved_contract_rows, key=lambda item: (item[0], item[1])):
                handle.write("\t".join(row) + "\n")

    print(
        f"opentitan LEC dropped-syntax summary: drop_remark_cases={len(drop_remark_case_rows)} pattern='{drop_remark_pattern}'",
        flush=True,
    )

    if failures:
        print(f"LEC failures: {failures}", file=sys.stderr)
        return 1
    print("LEC completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

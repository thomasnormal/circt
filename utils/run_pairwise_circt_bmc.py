#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run generic pairwise BMC checks with circt-bmc.

The case manifest is a tab-separated file with one case per line:

  case_id <TAB> top_module <TAB> source_files <TAB> include_dirs <TAB> case_path
          <TAB> timeout_secs <TAB> backend_mode <TAB> bmc_bound
          <TAB> ignore_asserts_until <TAB> assume_known_inputs
          <TAB> allow_multi_clock <TAB> bmc_extra_args
          <TAB> contract_source

Only the first three columns are required.

- source_files: ';'-separated list of source files.
- include_dirs: ';'-separated list of include directories (optional).
- case_path: logical case path written to output rows (optional).
- timeout_secs: per-case timeout override in seconds (optional).
- backend_mode: per-case backend override (optional):
  - default: inherit global env (BMC_RUN_SMTLIB/BMC_SMOKE_ONLY)
  - jit: force native/JIT backend (no --run-smtlib)
  - smtlib: force external SMT-LIB backend (--run-smtlib)
  - smoke: force smoke-only emit-mlir mode
- bmc_bound: per-case `-b` override (optional, non-negative integer; 0 maps to 1).
- ignore_asserts_until: per-case `--ignore-asserts-until` override (optional,
  non-negative integer).
- assume_known_inputs: per-case override for `--assume-known-inputs`:
  `default|on|off` (also accepts `1|0|true|false|yes|no`).
- allow_multi_clock: per-case override for `--allow-multi-clock`:
  `default|on|off` (also accepts `1|0|true|false|yes|no`).
- bmc_extra_args: optional shell-style per-case extra circt-bmc argument bundle.
  Restricted core options (module/bound/backend/known-input/multi-clock) are rejected
  to keep contract resolution deterministic.
- contract_source: optional provenance label for the case contract (for example
  `exact:aes_sbox_canright` or `pattern:re:^foo`) that is emitted in
  `--resolved-contracts-file` artifacts.
  The artifact writes `#resolved_contract_schema_version=1` on the first line
  and appends `contract_fingerprint` (stable sha256-derived digest) for drift
  checks.

Relative file paths are resolved against the manifest file directory.

ETXTBSY launch retry tuning:
- BMC_LAUNCH_ETXTBSY_RETRIES (default: 4)
- BMC_LAUNCH_ETXTBSY_BACKOFF_SECS (default: 0.2)
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    top_module: str
    source_files: list[str]
    include_dirs: list[str]
    case_path: str
    timeout_secs: int | None
    backend_mode: str
    bmc_bound: int | None
    ignore_asserts_until: int | None
    assume_known_inputs_mode: str
    allow_multi_clock_mode: str
    bmc_extra_args: list[str]
    contract_source: str


class TextFileBusyRetryExhausted(RuntimeError):
    def __init__(self, tool: str, attempts: int):
        super().__init__(
            f"runner_command_etxtbsy_retry_exhausted tool={tool} attempts={attempts}"
        )
        self.tool = tool
        self.attempts = attempts


def parse_nonnegative_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        print(f"invalid {name}: {raw}", file=sys.stderr)
        raise SystemExit(1)
    if value < 0:
        print(f"invalid {name}: {raw}", file=sys.stderr)
        raise SystemExit(1)
    return value


def parse_nonnegative_float(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        print(f"invalid {name}: {raw}", file=sys.stderr)
        raise SystemExit(1)
    if value < 0.0:
        print(f"invalid {name}: {raw}", file=sys.stderr)
        raise SystemExit(1)
    return value


def write_log(path: Path, stdout: str, stderr: str) -> None:
    data = ""
    if stdout:
        data += stdout
        if not data.endswith("\n"):
            data += "\n"
    if stderr:
        data += stderr
    path.write_text(data)


def parse_bmc_result(text: str) -> str | None:
    match = re.search(r"BMC_RESULT=(SAT|UNSAT|UNKNOWN)", text)
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


def normalize_error_reason(text: str) -> str:
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[^:\n]+:\d+(?::\d+)?:\s*", "", line)
        line = re.sub(r"^[Ee]rror:\s*", "", line)
        low = line.lower()
        if "text file busy" in low:
            return "runner_command_text_file_busy"
        if "no such file or directory" in low or "not found" in low:
            return "runner_command_not_found"
        if "permission denied" in low:
            return "runner_command_permission_denied"
        if "cannot allocate memory" in low or "memory exhausted" in low:
            return "command_oom"
        if "timed out" in low or "timeout" in low:
            return "command_timeout"
        line = re.sub(r"[0-9]+", "<n>", line)
        line = re.sub(r"[^A-Za-z0-9]+", "_", line).strip("_").lower()
        if not line:
            return "no_diag"
        if len(line) > 64:
            line = line[:64]
        return line
    return "no_diag"


def run_and_log(
    cmd: list[str],
    log_path: Path,
    out_path: Path | None,
    timeout_secs: int,
    etxtbsy_retries: int,
    etxtbsy_backoff_secs: float,
) -> subprocess.CompletedProcess[str]:
    # Binary relinking races can produce transient ETXTBSY while a tool is open
    # for writing; retry a few times with bounded backoff.
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(etxtbsy_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_secs if timeout_secs > 0 else None,
            )
            break
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            write_log(log_path, stdout, stderr)
            if out_path is not None:
                out_path.write_text(stdout)
            raise
        except OSError as exc:
            if exc.errno == errno.ETXTBSY and attempt < etxtbsy_retries:
                time.sleep(etxtbsy_backoff_secs * (attempt + 1))
                continue
            if exc.errno == errno.ETXTBSY:
                raise TextFileBusyRetryExhausted(cmd[0], attempt + 1) from exc
            raise
    if result is None:
        raise RuntimeError("internal error: subprocess result missing")
    write_log(log_path, result.stdout, result.stderr)
    if out_path is not None:
        out_path.write_text(result.stdout)
    return result


def append_exception_log(log_path: Path, stage: str, exc: Exception) -> None:
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        if log_path.exists() and log_path.stat().st_size > 0:
            handle.write("\n")
        handle.write(f"runner_command_exception stage={stage}\n")
        handle.write(trace)


def sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not sanitized:
        return "anon"
    if not re.match(r"[A-Za-z_]", sanitized[0]):
        sanitized = f"m_{sanitized}"
    return sanitized


def parse_semicolon_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(";") if item.strip()]


def parse_optional_nonnegative_int(
    raw: str, name: str, line_no: int
) -> int | None:
    token = raw.strip()
    if not token:
        return None
    try:
        value = int(token)
    except ValueError:
        print(
            (
                f"invalid cases file row {line_no}: {name} must be "
                f"non-negative integer, got '{raw}'"
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
    if value < 0:
        print(
            (
                f"invalid cases file row {line_no}: {name} must be "
                f"non-negative integer, got '{raw}'"
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value


def parse_case_backend(raw: str, line_no: int) -> str:
    token = raw.strip().lower()
    if not token:
        return "default"
    if token in {"default", "jit", "smtlib", "smoke"}:
        return token
    print(
        (
            f"invalid cases file row {line_no}: backend_mode must be one of "
            f"default|jit|smtlib|smoke, got '{raw}'"
        ),
        file=sys.stderr,
    )
    raise SystemExit(1)


def parse_case_toggle_mode(raw: str, line_no: int, name: str) -> str:
    token = raw.strip().lower()
    if not token:
        return "default"
    mapping = {
        "default": "default",
        "on": "on",
        "off": "off",
        "1": "on",
        "0": "off",
        "true": "on",
        "false": "off",
        "yes": "on",
        "no": "off",
    }
    if token in mapping:
        return mapping[token]
    print(
        (
            f"invalid cases file row {line_no}: {name} must be one of "
            f"default|on|off (or 1|0|true|false|yes|no), got '{raw}'"
        ),
        file=sys.stderr,
    )
    raise SystemExit(1)


def parse_case_bmc_extra_args(raw: str, line_no: int) -> list[str]:
    token = raw.strip()
    if not token:
        return []
    try:
        args = shlex.split(token)
    except ValueError as exc:
        print(
            f"invalid cases file row {line_no}: bmc_extra_args parse error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    restricted = (
        "--module",
        "-b",
        "--bound",
        "--ignore-asserts-until",
        "--emit-mlir",
        "--run-smtlib",
        "--z3-path",
        "--shared-libs",
        "--assume-known-inputs",
        "--allow-multi-clock",
    )
    for arg in args:
        for key in restricted:
            if arg == key or arg.startswith(f"{key}="):
                print(
                    (
                        f"invalid cases file row {line_no}: bmc_extra_args contains "
                        f"restricted option '{arg}'"
                    ),
                    file=sys.stderr,
                )
                raise SystemExit(1)
    return args


def compute_contract_fingerprint(fields: list[str]) -> str:
    payload = "\x1f".join(fields).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_cases(cases_file: Path) -> list[CaseSpec]:
    base_dir = cases_file.parent
    cases: list[CaseSpec] = []
    with cases_file.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                print(
                    f"invalid cases file row {line_no}: expected >=3 tab columns",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            case_id = parts[0].strip()
            top_module = parts[1].strip()
            source_files_raw = parts[2].strip()
            include_dirs_raw = parts[3].strip() if len(parts) > 3 else ""
            case_path_raw = parts[4].strip() if len(parts) > 4 else ""
            timeout_secs_raw = parts[5].strip() if len(parts) > 5 else ""
            backend_mode_raw = parts[6].strip() if len(parts) > 6 else ""
            bmc_bound_raw = parts[7].strip() if len(parts) > 7 else ""
            ignore_asserts_until_raw = parts[8].strip() if len(parts) > 8 else ""
            assume_known_inputs_raw = parts[9].strip() if len(parts) > 9 else ""
            allow_multi_clock_raw = parts[10].strip() if len(parts) > 10 else ""
            bmc_extra_args_raw = parts[11].strip() if len(parts) > 11 else ""
            contract_source_raw = parts[12].strip() if len(parts) > 12 else ""
            if not case_id or not top_module:
                print(
                    f"invalid cases file row {line_no}: empty case_id/top_module",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            source_files = parse_semicolon_list(source_files_raw)
            if not source_files:
                print(
                    f"invalid cases file row {line_no}: no source files",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            include_dirs = parse_semicolon_list(include_dirs_raw)
            timeout_secs = parse_optional_nonnegative_int(
                timeout_secs_raw, "timeout_secs", line_no
            )
            backend_mode = parse_case_backend(backend_mode_raw, line_no)
            bmc_bound = parse_optional_nonnegative_int(
                bmc_bound_raw, "bmc_bound", line_no
            )
            ignore_asserts_until = parse_optional_nonnegative_int(
                ignore_asserts_until_raw, "ignore_asserts_until", line_no
            )
            assume_known_inputs_mode = parse_case_toggle_mode(
                assume_known_inputs_raw, line_no, "assume_known_inputs"
            )
            allow_multi_clock_mode = parse_case_toggle_mode(
                allow_multi_clock_raw, line_no, "allow_multi_clock"
            )
            bmc_extra_args = parse_case_bmc_extra_args(
                bmc_extra_args_raw, line_no
            )

            def resolve_path(path_raw: str) -> str:
                path = Path(path_raw)
                if not path.is_absolute():
                    path = base_dir / path
                return str(path.resolve())

            cases.append(
                CaseSpec(
                    case_id=case_id,
                    top_module=top_module,
                    source_files=[resolve_path(path) for path in source_files],
                    include_dirs=[resolve_path(path) for path in include_dirs],
                    case_path=case_path_raw,
                    timeout_secs=timeout_secs,
                    backend_mode=backend_mode,
                    bmc_bound=bmc_bound,
                    ignore_asserts_until=ignore_asserts_until,
                    assume_known_inputs_mode=assume_known_inputs_mode,
                    allow_multi_clock_mode=allow_multi_clock_mode,
                    bmc_extra_args=bmc_extra_args,
                    contract_source=contract_source_raw,
                )
            )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pairwise BMC checks from a case manifest."
    )
    parser.add_argument(
        "--cases-file",
        required=True,
        help="Path to tab-separated case manifest.",
    )
    parser.add_argument(
        "--suite-name",
        default=os.environ.get("BMC_SUITE_NAME", "pairwise"),
        help="Suite name recorded in results output (default: pairwise).",
    )
    parser.add_argument(
        "--mode-label",
        default=os.environ.get("BMC_MODE_LABEL", "BMC"),
        help="Mode label recorded in results output (default: BMC).",
    )
    parser.add_argument(
        "--bound",
        default=os.environ.get("BOUND", "1"),
        help="BMC bound (default: env BOUND or 1).",
    )
    parser.add_argument(
        "--ignore-asserts-until",
        default=os.environ.get("IGNORE_ASSERTS_UNTIL", "0"),
        help="BMC --ignore-asserts-until value (default: env or 0).",
    )
    parser.add_argument(
        "--include-dir",
        action="append",
        default=[],
        help="Global include directory (repeatable).",
    )
    parser.add_argument(
        "--workdir",
        default="",
        help="Optional work directory (default: temp directory).",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep work directory after run.",
    )
    parser.add_argument(
        "--results-file",
        default=os.environ.get("OUT", ""),
        help="Optional TSV output path for per-case rows.",
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
    args = parser.parse_args()

    cases_file = Path(args.cases_file).resolve()
    if not cases_file.is_file():
        print(f"cases file not found: {cases_file}", file=sys.stderr)
        return 1
    cases = load_cases(cases_file)
    if not cases:
        print("No pairwise BMC cases selected.", file=sys.stderr)
        return 1

    circt_verilog = os.environ.get("CIRCT_VERILOG", "build/bin/circt-verilog")
    circt_verilog_args = shlex.split(os.environ.get("CIRCT_VERILOG_ARGS", ""))
    circt_opt = os.environ.get("CIRCT_OPT", "build/bin/circt-opt")
    circt_opt_args = shlex.split(os.environ.get("CIRCT_OPT_ARGS", ""))
    circt_bmc = os.environ.get("CIRCT_BMC", "build/bin/circt-bmc")
    circt_bmc_args = shlex.split(os.environ.get("CIRCT_BMC_ARGS", ""))
    z3_lib = os.environ.get("Z3_LIB", str(Path.home() / "z3-install/lib64/libz3.so"))
    bmc_run_smtlib = os.environ.get("BMC_RUN_SMTLIB", "1") == "1"
    bmc_smoke_only = os.environ.get("BMC_SMOKE_ONLY", "0") == "1"
    bmc_assume_known_inputs = os.environ.get("BMC_ASSUME_KNOWN_INPUTS", "0") == "1"
    bmc_allow_multi_clock = os.environ.get("BMC_ALLOW_MULTI_CLOCK", "0") == "1"
    bmc_prepare_core_with_circt_opt = (
        os.environ.get("BMC_PREPARE_CORE_WITH_CIRCT_OPT", "1") == "1"
    )
    bmc_prepare_core_passes = shlex.split(
        os.environ.get(
            "BMC_PREPARE_CORE_PASSES",
            "--lower-lec-llvm --reconcile-unrealized-casts",
        )
    )
    drop_remark_pattern = os.environ.get(
        "BMC_DROP_REMARK_PATTERN",
        os.environ.get("DROP_REMARK_PATTERN", "will be dropped during lowering"),
    )
    timeout_secs = parse_nonnegative_int(
        os.environ.get("CIRCT_TIMEOUT_SECS", "300"), "CIRCT_TIMEOUT_SECS"
    )
    etxtbsy_retries = parse_nonnegative_int(
        os.environ.get("BMC_LAUNCH_ETXTBSY_RETRIES", "4"),
        "BMC_LAUNCH_ETXTBSY_RETRIES",
    )
    etxtbsy_backoff_secs = parse_nonnegative_float(
        os.environ.get("BMC_LAUNCH_ETXTBSY_BACKOFF_SECS", "0.2"),
        "BMC_LAUNCH_ETXTBSY_BACKOFF_SECS",
    )
    bound = parse_nonnegative_int(args.bound, "--bound")
    if bound == 0:
        bound = 1
    ignore_asserts_until = parse_nonnegative_int(
        args.ignore_asserts_until, "--ignore-asserts-until"
    )

    global_include_dirs = [str(Path(path).resolve()) for path in args.include_dir]
    mode_label = args.mode_label.strip() or "BMC"
    suite_name = args.suite_name.strip() or "pairwise"

    def resolve_case_execution_profile(case: CaseSpec) -> tuple[bool, bool]:
        # Returns (smoke_only, run_smtlib) for the case.
        if case.backend_mode == "smoke":
            return True, False
        if case.backend_mode == "smtlib":
            return False, True
        if case.backend_mode == "jit":
            return False, False
        return bmc_smoke_only, bmc_run_smtlib

    def resolve_case_toggle(mode: str, inherited: bool) -> bool:
        if mode == "on":
            return True
        if mode == "off":
            return False
        return inherited

    requires_smtlib = any(
        (not smoke_only) and run_smtlib
        for smoke_only, run_smtlib in (
            resolve_case_execution_profile(case) for case in cases
        )
    )

    z3_bin = os.environ.get("Z3_BIN", "")
    if requires_smtlib:
        if not z3_bin:
            z3_bin = shutil.which("z3") or ""
        if not z3_bin and Path.home().joinpath("z3-install/bin/z3").is_file():
            z3_bin = str(Path.home() / "z3-install/bin/z3")
        if not z3_bin and Path.home().joinpath("z3/build/z3").is_file():
            z3_bin = str(Path.home() / "z3/build/z3")
        if not z3_bin:
            print("z3 not found; set Z3_BIN or disable BMC_RUN_SMTLIB", file=sys.stderr)
            return 1

    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        keep_workdir = True
    else:
        workdir = Path(tempfile.mkdtemp(prefix="pairwise-bmc-"))
        keep_workdir = args.keep_workdir

    rows: list[tuple[str, ...]] = []
    drop_remark_case_rows: list[tuple[str, str]] = []
    drop_remark_reason_rows: list[tuple[str, str, str]] = []
    timeout_reason_rows: list[tuple[str, str, str]] = []
    resolved_contract_rows: list[tuple[str, ...]] = []
    drop_remark_seen_case_reasons: set[tuple[str, str]] = set()
    timeout_reason_seen: set[tuple[str, str]] = set()

    passed = 0
    failed = 0
    errored = 0
    unknown = 0
    timeout = 0
    total = 0
    try:
        print(f"Running BMC on {len(cases)} pairwise case(s)...", flush=True)
        for case in cases:
            total += 1
            case_dir = workdir / sanitize_identifier(case.case_id)
            case_dir.mkdir(parents=True, exist_ok=True)
            case_path = case.case_path or str(case_dir)
            case_smoke_only, case_run_smtlib = resolve_case_execution_profile(case)
            case_timeout_secs = (
                timeout_secs if case.timeout_secs is None else case.timeout_secs
            )
            case_bound = bound if case.bmc_bound is None else case.bmc_bound
            if case_bound == 0:
                case_bound = 1
            case_ignore_asserts_until = (
                ignore_asserts_until
                if case.ignore_asserts_until is None
                else case.ignore_asserts_until
            )
            case_assume_known_inputs = resolve_case_toggle(
                case.assume_known_inputs_mode, bmc_assume_known_inputs
            )
            case_allow_multi_clock = resolve_case_toggle(
                case.allow_multi_clock_mode, bmc_allow_multi_clock
            )
            case_bmc_extra_args_text = shlex.join(case.bmc_extra_args)
            contract_source = case.contract_source.strip() or "manifest"
            contract_fields = [
                contract_source,
                case.backend_mode,
                "1" if case_smoke_only else "0",
                "1" if case_run_smtlib else "0",
                str(case_timeout_secs),
                str(case_bound),
                str(case_ignore_asserts_until),
                "1" if case_assume_known_inputs else "0",
                "1" if case_allow_multi_clock else "0",
                case_bmc_extra_args_text,
            ]
            contract_fingerprint = compute_contract_fingerprint(contract_fields)
            resolved_contract_rows.append(
                (
                    case.case_id,
                    case_path,
                    *contract_fields,
                    contract_fingerprint,
                )
            )

            verilog_log_path = case_dir / "circt-verilog.log"
            opt_log_path = case_dir / "circt-opt.log"
            bmc_log_path = case_dir / "circt-bmc.log"
            bmc_out_path = case_dir / "circt-bmc.out"
            out_mlir = case_dir / "pairwise_bmc.mlir"
            prepped_mlir = case_dir / "pairwise_bmc.prepared.mlir"

            include_dirs = global_include_dirs + case.include_dirs
            verilog_cmd = [
                circt_verilog,
                "--ir-hw",
                "-o",
                str(out_mlir),
                "--single-unit",
                "--no-uvm-auto-include",
                f"--top={case.top_module}",
            ]
            for include_dir in include_dirs:
                verilog_cmd += ["-I", include_dir]
            verilog_cmd += circt_verilog_args
            verilog_cmd += case.source_files

            opt_cmd = [circt_opt, str(out_mlir)]
            opt_cmd += bmc_prepare_core_passes
            opt_cmd += circt_opt_args
            opt_cmd += ["-o", str(prepped_mlir)]

            bmc_cmd = [
                circt_bmc,
                str(prepped_mlir if bmc_prepare_core_with_circt_opt else out_mlir),
                f"--module={case.top_module}",
                "-b",
                str(case_bound),
                f"--ignore-asserts-until={case_ignore_asserts_until}",
            ]
            if case_smoke_only:
                bmc_cmd.append("--emit-mlir")
            else:
                if case_run_smtlib:
                    bmc_cmd.append("--run-smtlib")
                    bmc_cmd.append(f"--z3-path={z3_bin}")
                elif z3_lib:
                    bmc_cmd.append(f"--shared-libs={z3_lib}")
            if case_assume_known_inputs:
                bmc_cmd.append("--assume-known-inputs")
            if case_allow_multi_clock:
                bmc_cmd.append("--allow-multi-clock")
            bmc_cmd += case.bmc_extra_args
            bmc_cmd += circt_bmc_args

            stage = "verilog"
            try:
                verilog_result = run_and_log(
                    verilog_cmd,
                    verilog_log_path,
                    None,
                    case_timeout_secs,
                    etxtbsy_retries,
                    etxtbsy_backoff_secs,
                )
                if verilog_result.returncode != 0:
                    reason = normalize_error_reason(verilog_log_path.read_text())
                    rows.append(
                        (
                            "ERROR",
                            case.case_id,
                            case_path,
                            suite_name,
                            mode_label,
                            "CIRCT_VERILOG_ERROR",
                            reason,
                        )
                    )
                    errored += 1
                    print(f"{case.case_id:32} ERROR (CIRCT_VERILOG_ERROR)", flush=True)
                    continue

                reasons = extract_drop_reasons(
                    verilog_log_path.read_text(), drop_remark_pattern
                )
                if reasons:
                    drop_remark_case_rows.append((case.case_id, case_path))
                    for reason in reasons:
                        key = (case.case_id, reason)
                        if key in drop_remark_seen_case_reasons:
                            continue
                        drop_remark_seen_case_reasons.add(key)
                        drop_remark_reason_rows.append((case.case_id, case_path, reason))

                if bmc_prepare_core_with_circt_opt:
                    stage = "opt"
                    opt_result = run_and_log(
                        opt_cmd,
                        opt_log_path,
                        None,
                        case_timeout_secs,
                        etxtbsy_retries,
                        etxtbsy_backoff_secs,
                    )
                    if opt_result.returncode != 0:
                        reason = normalize_error_reason(opt_log_path.read_text())
                        rows.append(
                            (
                                "ERROR",
                                case.case_id,
                                case_path,
                                suite_name,
                                mode_label,
                                "CIRCT_OPT_ERROR",
                                reason,
                            )
                        )
                        errored += 1
                        print(f"{case.case_id:32} ERROR (CIRCT_OPT_ERROR)", flush=True)
                        continue

                stage = "bmc"
                bmc_result = run_and_log(
                    bmc_cmd,
                    bmc_log_path,
                    bmc_out_path,
                    case_timeout_secs,
                    etxtbsy_retries,
                    etxtbsy_backoff_secs,
                )
                combined = bmc_log_path.read_text() + "\n" + bmc_out_path.read_text()
                bmc_tag = parse_bmc_result(combined)

                if case_smoke_only:
                    if bmc_result.returncode == 0:
                        rows.append(
                            (
                                "PASS",
                                case.case_id,
                                case_path,
                                suite_name,
                                mode_label,
                                "SMOKE_ONLY",
                            )
                        )
                        passed += 1
                        print(f"{case.case_id:32} OK", flush=True)
                    else:
                        reason = normalize_error_reason(combined)
                        rows.append(
                            (
                                "ERROR",
                                case.case_id,
                                case_path,
                                suite_name,
                                mode_label,
                                "CIRCT_BMC_ERROR",
                                reason,
                            )
                        )
                        errored += 1
                        print(f"{case.case_id:32} ERROR (CIRCT_BMC_ERROR)", flush=True)
                    continue

                if bmc_tag == "UNSAT":
                    rows.append(
                        (
                            "PASS",
                            case.case_id,
                            case_path,
                            suite_name,
                            mode_label,
                            "UNSAT",
                        )
                    )
                    passed += 1
                    print(f"{case.case_id:32} OK", flush=True)
                elif bmc_tag == "SAT":
                    rows.append(
                        (
                            "FAIL",
                            case.case_id,
                            f"{case_path}#SAT",
                            suite_name,
                            mode_label,
                            "SAT",
                        )
                    )
                    failed += 1
                    print(f"{case.case_id:32} FAIL (SAT)", flush=True)
                elif bmc_tag == "UNKNOWN":
                    rows.append(
                        (
                            "UNKNOWN",
                            case.case_id,
                            f"{case_path}#UNKNOWN",
                            suite_name,
                            mode_label,
                            "UNKNOWN",
                        )
                    )
                    unknown += 1
                    print(f"{case.case_id:32} UNKNOWN", flush=True)
                else:
                    reason = normalize_error_reason(combined)
                    rows.append(
                        (
                            "ERROR",
                            case.case_id,
                            case_path,
                            suite_name,
                            mode_label,
                            "CIRCT_BMC_ERROR",
                            reason,
                        )
                    )
                    errored += 1
                    print(f"{case.case_id:32} ERROR (CIRCT_BMC_ERROR)", flush=True)
            except subprocess.TimeoutExpired:
                if stage == "verilog":
                    timeout_diag = "CIRCT_VERILOG_TIMEOUT"
                    timeout_reason = "frontend_command_timeout"
                elif stage == "opt":
                    timeout_diag = "CIRCT_OPT_TIMEOUT"
                    timeout_reason = "frontend_command_timeout"
                else:
                    timeout_diag = "BMC_TIMEOUT"
                    timeout_reason = "solver_command_timeout"
                key = (case.case_id, timeout_reason)
                if key not in timeout_reason_seen:
                    timeout_reason_seen.add(key)
                    timeout_reason_rows.append((case.case_id, case_path, timeout_reason))
                rows.append(
                    (
                        "TIMEOUT",
                        case.case_id,
                        case_path,
                        suite_name,
                        mode_label,
                        timeout_diag,
                        timeout_reason,
                    )
                )
                timeout += 1
                print(f"{case.case_id:32} TIMEOUT ({timeout_diag})", flush=True)
            except Exception as exc:
                if stage == "verilog":
                    error_diag = "CIRCT_VERILOG_ERROR"
                    error_log_path = verilog_log_path
                elif stage == "opt":
                    error_diag = "CIRCT_OPT_ERROR"
                    error_log_path = opt_log_path
                else:
                    error_diag = "CIRCT_BMC_ERROR"
                    error_log_path = bmc_log_path
                append_exception_log(error_log_path, stage, exc)
                if isinstance(exc, TextFileBusyRetryExhausted):
                    reason = (
                        f"runner_command_exception_{stage}_"
                        "runner_command_etxtbsy_retry_exhausted"
                    )
                else:
                    reason_body = normalize_error_reason(
                        f"{type(exc).__name__}: {exc}"
                    )
                    reason = (
                        "runner_command_exception"
                        if reason_body == "no_diag"
                        else f"runner_command_exception_{stage}_{reason_body}"
                    )
                rows.append(
                    (
                        "ERROR",
                        case.case_id,
                        case_path,
                        suite_name,
                        mode_label,
                        error_diag,
                        reason,
                    )
                )
                errored += 1
                print(f"{case.case_id:32} ERROR ({error_diag})", flush=True)
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    if args.results_file:
        results_path = Path(args.results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as handle:
            for row in sorted(rows, key=lambda item: (item[1], item[0], item[2])):
                handle.write("\t".join(row) + "\n")
        print(f"results: {results_path}", flush=True)

    if args.drop_remark_cases_file:
        case_path = Path(args.drop_remark_cases_file)
        case_path.parent.mkdir(parents=True, exist_ok=True)
        with case_path.open("w", encoding="utf-8") as handle:
            for row in sorted(set(drop_remark_case_rows), key=lambda item: item[0]):
                handle.write("\t".join(row) + "\n")

    if args.drop_remark_reasons_file:
        reason_path = Path(args.drop_remark_reasons_file)
        reason_path.parent.mkdir(parents=True, exist_ok=True)
        with reason_path.open("w", encoding="utf-8") as handle:
            for row in sorted(drop_remark_reason_rows, key=lambda item: (item[0], item[2])):
                handle.write("\t".join(row) + "\n")

    if args.timeout_reasons_file:
        timeout_path = Path(args.timeout_reasons_file)
        timeout_path.parent.mkdir(parents=True, exist_ok=True)
        with timeout_path.open("w", encoding="utf-8") as handle:
            for row in sorted(timeout_reason_rows, key=lambda item: (item[0], item[2])):
                handle.write("\t".join(row) + "\n")

    if args.resolved_contracts_file:
        contracts_path = Path(args.resolved_contracts_file)
        contracts_path.parent.mkdir(parents=True, exist_ok=True)
        with contracts_path.open("w", encoding="utf-8") as handle:
            handle.write("#resolved_contract_schema_version=1\n")
            for row in sorted(resolved_contract_rows, key=lambda item: (item[0], item[1])):
                handle.write("\t".join(row) + "\n")

    print(
        f"{suite_name} BMC dropped-syntax summary: "
        f"drop_remark_cases={len(set(drop_remark_case_rows))} "
        f"pattern='{drop_remark_pattern}'",
        flush=True,
    )
    print(
        f"{suite_name} BMC summary: total={total} pass={passed} fail={failed} "
        f"xfail=0 xpass=0 error={errored} skip=0 unknown={unknown} timeout={timeout}",
        flush=True,
    )

    if failed or errored or unknown or timeout:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

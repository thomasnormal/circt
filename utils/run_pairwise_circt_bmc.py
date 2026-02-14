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
          <TAB> contract_source <TAB> verilog_defines

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
- verilog_defines: optional ';'-separated preprocessor define list forwarded
  to `circt-verilog` as `-D<token>` arguments.
  The artifact writes `#resolved_contract_schema_version=1` on the first line
  and appends `contract_fingerprint` (stable sha256-derived digest) for drift
  checks.

Relative file paths are resolved against the manifest file directory.

ETXTBSY launch retry tuning:
- BMC_LAUNCH_ETXTBSY_RETRIES (default: 4)
- BMC_LAUNCH_ETXTBSY_BACKOFF_SECS (default: 0.2)

General launch retry tuning:
- BMC_LAUNCH_RETRY_ATTEMPTS (default: 4)
- BMC_LAUNCH_RETRY_BACKOFF_SECS (default: 0.2)
- BMC_LAUNCH_RETRYABLE_EXIT_CODES (default: 126,127)
- BMC_LAUNCH_COPY_FALLBACK (default: 0)
- BMC_LAUNCH_EVENTS_OUT (optional TSV output)

Verilog frontend mode:
- BMC_VERILOG_SINGLE_UNIT_MODE (default: auto)
  Values:
  - auto: start with `--single-unit`; retry once without it for known
    macro-preprocessor failures.
  - on: always use `--single-unit`.
  - off: never use `--single-unit`.
- BMC_VERILOG_XILINX_PRIMITIVE_STUB_MODE (default: auto)
  Values:
  - auto: on known unknown-module failures for common Xilinx clock/pad
    primitives, retry once with shim module definitions.
  - on: always include shim module definitions.
  - off: disable shim module retry/injection.
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
    verilog_defines: list[str]


@dataclass(frozen=True)
class AssertionSite:
    ordinal: int
    line_index: int
    line_text: str


@dataclass(frozen=True)
class CoverSite:
    ordinal: int
    line_index: int
    line_text: str


class TextFileBusyRetryExhausted(RuntimeError):
    def __init__(self, tool: str, attempts: int):
        super().__init__(
            f"runner_command_etxtbsy_retry_exhausted tool={tool} attempts={attempts}"
        )
        self.tool = tool
        self.attempts = attempts


RETRYABLE_LAUNCH_PATTERNS = (
    "text file busy",
    "etxtbsy",
    "permission denied",
    "posix_spawn failed",
    "resource temporarily unavailable",
)

UNKNOWN_MODULE_PATTERNS = (
    r"unknown module ['\"]([^'\"]+)['\"]",
    r"unknown module [`]([^`]+)[`]",
    r"unknown module \\([^\\]+)\\",
    r"unknown module ([A-Za-z_][A-Za-z0-9_$.:/-]*)",
)

XILINX_PRIMITIVE_STUB_MODULES = frozenset(
    {
        "BUFG",
        "BUFGCE",
        "BUFGCE_DIV",
        "BUFGCTRL",
        "BUFGMUX",
        "BUFR",
        "IBUF_IBUFDISABLE",
        "IOBUF",
    }
)


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


def parse_exit_codes(raw: str, name: str) -> set[int]:
    value = raw.strip()
    if not value:
        return set()
    out: set[int] = set()
    for token in value.split(","):
        piece = token.strip()
        if not piece:
            continue
        try:
            code = int(piece)
        except ValueError:
            print(f"invalid {name}: {raw}", file=sys.stderr)
            raise SystemExit(1)
        if code < 0:
            print(f"invalid {name}: {raw}", file=sys.stderr)
            raise SystemExit(1)
        out.add(code)
    return out


def is_retryable_launch_failure_output(stdout: str, stderr: str) -> bool:
    lowered = f"{stdout}\n{stderr}".lower()
    return any(pattern in lowered for pattern in RETRYABLE_LAUNCH_PATTERNS)


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


def parse_assertion_status(text: str) -> str | None:
    match = re.search(r"BMC_ASSERTION_STATUS=(PROVEN|FAILING|VACUOUS|UNKNOWN)", text)
    if not match:
        return None
    return match.group(1)


def collect_candidate_assertion_sites(lines: list[str]) -> list[AssertionSite]:
    sites: list[AssertionSite] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if "verif.assert" not in stripped:
            continue
        if "{bmc.final}" in stripped:
            continue
        sites.append(
            AssertionSite(
                ordinal=len(sites),
                line_index=idx,
                line_text=stripped,
            )
        )
    return sites


def collect_candidate_cover_sites(lines: list[str]) -> list[CoverSite]:
    sites: list[CoverSite] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if "verif.cover" not in stripped:
            continue
        sites.append(
            CoverSite(
                ordinal=len(sites),
                line_index=idx,
                line_text=stripped,
            )
        )
    return sites


def build_isolated_assertion_mlir(
    lines: list[str], sites: list[AssertionSite], keep_ordinal: int
) -> str:
    isolated = list(lines)
    for site in sites:
        if site.ordinal == keep_ordinal:
            continue
        isolated[site.line_index] = isolated[site.line_index].replace(
            "verif.assert", "verif.assume", 1
        )
    return "\n".join(isolated) + "\n"


def build_isolated_cover_mlir(
    lines: list[str],
    assertion_sites: list[AssertionSite],
    cover_sites: list[CoverSite],
    keep_ordinal: int,
) -> str:
    isolated = list(lines)
    for site in assertion_sites:
        isolated[site.line_index] = isolated[site.line_index].replace(
            "verif.assert", "verif.assume", 1
        )
    for site in cover_sites:
        if site.ordinal == keep_ordinal:
            continue
        prefix = re.match(r"^\s*", isolated[site.line_index]).group(0)
        isolated[site.line_index] = f"{prefix}// cover-granular-disabled"
    return "\n".join(isolated) + "\n"


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


def is_single_unit_retryable_preprocessor_failure(log_text: str) -> bool:
    low = log_text.lower()
    return (
        "macro operators may only be used within a macro definition" in low
        or "unexpected conditional directive" in low
    )


def extract_unknown_modules(log_text: str) -> set[str]:
    modules: set[str] = set()
    for line in log_text.splitlines():
        for pattern in UNKNOWN_MODULE_PATTERNS:
            match = re.search(pattern, line)
            if not match:
                continue
            module = match.group(1).strip()
            if module:
                modules.add(module)
    return modules


def is_xilinx_primitive_stub_retryable_failure(log_text: str) -> bool:
    unknown_modules = extract_unknown_modules(log_text)
    if not unknown_modules:
        return False
    return any(mod in XILINX_PRIMITIVE_STUB_MODULES for mod in unknown_modules)


def write_xilinx_primitive_stub_file(path: Path) -> None:
    path.write_text(
        """// Auto-generated fallback shims for common Xilinx primitives used in
// clock/pad wrappers. This preserves conservative, deterministic behavior for
// formal frontend ingestion when vendor primitive libraries are unavailable.
module BUFG(input I, output O);
  assign O = I;
endmodule

module BUFR(input I, output O);
  assign O = I;
endmodule

module BUFGMUX(input S, input I0, input I1, output O);
  assign O = S ? I1 : I0;
endmodule

module BUFGCE #(
  parameter string SIM_DEVICE = "ULTRASCALE",
  parameter bit IS_I_INVERTED = 1'b0
) (
  input I,
  input CE,
  output O
);
  wire i_eff = IS_I_INVERTED ? ~I : I;
  assign O = CE ? i_eff : 1'b0;
endmodule

module BUFGCE_DIV #(
  parameter integer BUFGCE_DIVIDE = 1,
  parameter bit IS_CLR_INVERTED = 1'b0
) (
  input I,
  input CE,
  input CLR,
  output O
);
  wire clr_eff = IS_CLR_INVERTED ? ~CLR : CLR;
  assign O = clr_eff ? 1'b0 : (CE ? I : 1'b0);
endmodule

module BUFGCTRL #(
  parameter bit INIT_OUT = 1'b0,
  parameter bit IS_I0_INVERTED = 1'b0,
  parameter bit IS_S0_INVERTED = 1'b0
) (
  input I0,
  input I1,
  input CE0,
  input CE1,
  input IGNORE0 = 1'b0,
  input IGNORE1 = 1'b0,
  input S0,
  input S1,
  output O
);
  wire i0_eff = IS_I0_INVERTED ? ~I0 : I0;
  wire s0_eff = IS_S0_INVERTED ? ~S0 : S0;
  wire sel0 = (s0_eff & CE0) | IGNORE0;
  wire sel1 = (S1 & CE1) | IGNORE1;
  assign O = sel0 ? i0_eff : (sel1 ? I1 : INIT_OUT);
endmodule

module IBUF_IBUFDISABLE(
  input I,
  input IBUFDISABLE,
  output O
);
  assign O = IBUFDISABLE ? 1'b0 : I;
endmodule

module IOBUF(
  input T,
  input I,
  output O,
  inout IO
);
  assign IO = T ? 1'bz : I;
  assign O = IO;
endmodule
""",
        encoding="utf-8",
    )


def run_and_log(
    cmd: list[str],
    log_path: Path,
    out_path: Path | None,
    timeout_secs: int,
    etxtbsy_retries: int,
    etxtbsy_backoff_secs: float,
    launch_retry_attempts: int,
    launch_retry_backoff_secs: float,
    launch_retryable_exit_codes: set[int],
    launch_copy_fallback: bool,
    launch_event_rows: list[tuple[str, ...]] | None = None,
    case_id: str = "",
    case_path: str = "",
    stage_label: str = "",
) -> subprocess.CompletedProcess[str]:
    # Binary relinking races can produce transient ETXTBSY while a tool is open
    # for writing; retry a few times with bounded backoff.
    active_cmd = list(cmd)
    launch_retry_count = 0
    etxtbsy_retry_count = 0
    launch_copy_fallback_used = False
    retry_notes: list[str] = []

    def append_launch_event(
        event_kind: str,
        reason: str,
        attempt: int | None,
        delay_secs: float | None,
        exit_code: int | None,
        fallback_tool: str,
    ) -> None:
        if launch_event_rows is None:
            return
        launch_event_rows.append(
            (
                event_kind,
                case_id,
                case_path,
                stage_label,
                active_cmd[0] if active_cmd else "",
                reason,
                str(attempt) if attempt is not None else "",
                f"{delay_secs:.3f}" if delay_secs is not None else "",
                str(exit_code) if exit_code is not None else "",
                fallback_tool,
            )
        )

    result: subprocess.CompletedProcess[str] | None = None
    while True:
        try:
            result = subprocess.run(
                active_cmd,
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
        except OSError as exc:
            if exc.errno == errno.ETXTBSY:
                if etxtbsy_retry_count < etxtbsy_retries:
                    etxtbsy_retry_count += 1
                    delay = etxtbsy_backoff_secs * etxtbsy_retry_count
                    retry_notes.append(
                        (
                            "runner_command_launch_retry "
                            f"reason=etxtbsy attempt={etxtbsy_retry_count} "
                            f"delay_secs={delay:.3f}"
                        )
                    )
                    append_launch_event(
                        "RETRY",
                        "etxtbsy",
                        etxtbsy_retry_count,
                        delay,
                        None,
                        "",
                    )
                    time.sleep(delay)
                    continue
                if launch_copy_fallback and not launch_copy_fallback_used:
                    tool_path = Path(active_cmd[0])
                    if tool_path.is_file():
                        fallback_tool = (
                            log_path.parent / f"{tool_path.name}.launch-fallback"
                        )
                        shutil.copy2(tool_path, fallback_tool)
                        fallback_tool.chmod(fallback_tool.stat().st_mode | 0o111)
                        active_cmd[0] = str(fallback_tool)
                        launch_copy_fallback_used = True
                        launch_retry_count = 0
                        etxtbsy_retry_count = 0
                        retry_notes.append(
                            (
                                "runner_command_launch_fallback "
                                f"tool={tool_path} fallback={fallback_tool}"
                            )
                        )
                        append_launch_event(
                            "FALLBACK",
                            "etxtbsy_retry_exhausted",
                            None,
                            None,
                            None,
                            str(fallback_tool),
                        )
                        continue
                raise TextFileBusyRetryExhausted(
                    active_cmd[0], etxtbsy_retry_count + 1
                ) from exc
            raise
        if (
            result.returncode != 0
            and result.returncode in launch_retryable_exit_codes
            and is_retryable_launch_failure_output(result.stdout, result.stderr)
        ):
            if launch_retry_count < launch_retry_attempts:
                launch_retry_count += 1
                delay = launch_retry_backoff_secs * launch_retry_count
                retry_notes.append(
                    (
                        "runner_command_launch_retry "
                        f"reason=retryable_exit_code_{result.returncode} "
                        f"attempt={launch_retry_count} delay_secs={delay:.3f}"
                    )
                )
                append_launch_event(
                    "RETRY",
                    f"retryable_exit_code_{result.returncode}",
                    launch_retry_count,
                    delay,
                    result.returncode,
                    "",
                )
                time.sleep(delay)
                continue
            if launch_copy_fallback and not launch_copy_fallback_used:
                tool_path = Path(active_cmd[0])
                if tool_path.is_file():
                    fallback_tool = log_path.parent / f"{tool_path.name}.launch-fallback"
                    shutil.copy2(tool_path, fallback_tool)
                    fallback_tool.chmod(fallback_tool.stat().st_mode | 0o111)
                    active_cmd[0] = str(fallback_tool)
                    launch_copy_fallback_used = True
                    launch_retry_count = 0
                    etxtbsy_retry_count = 0
                    retry_notes.append(
                        (
                            "runner_command_launch_fallback "
                            f"tool={tool_path} fallback={fallback_tool}"
                        )
                    )
                    append_launch_event(
                        "FALLBACK",
                        f"retryable_exit_code_{result.returncode}_retry_exhausted",
                        None,
                        None,
                        result.returncode,
                        str(fallback_tool),
                    )
                    continue
        break
    if result is None:
        raise RuntimeError("internal error: subprocess result missing")
    write_log(log_path, result.stdout, result.stderr)
    if retry_notes:
        with log_path.open("a", encoding="utf-8") as handle:
            if result.stdout or result.stderr:
                handle.write("\n")
            for note in retry_notes:
                handle.write(note)
                handle.write("\n")
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


def parse_case_verilog_defines(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(";") if item.strip()]


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
            verilog_defines_raw = parts[13].strip() if len(parts) > 13 else ""
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
                    verilog_defines=parse_case_verilog_defines(
                        verilog_defines_raw
                    ),
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
    parser.add_argument(
        "--assertion-results-file",
        default=os.environ.get("BMC_ASSERTION_RESULTS_OUT", ""),
        help="Optional TSV output path for per-assertion BMC rows.",
    )
    parser.add_argument(
        "--cover-results-file",
        default=os.environ.get("BMC_COVER_RESULTS_OUT", ""),
        help="Optional TSV output path for per-cover BMC rows.",
    )
    parser.add_argument(
        "--launch-events-file",
        default=os.environ.get("BMC_LAUNCH_EVENTS_OUT", ""),
        help="Optional TSV output path for launch retry/fallback events.",
    )
    parser.add_argument(
        "--assertion-granular",
        action="store_true",
        default=os.environ.get("BMC_ASSERTION_GRANULAR", "0") == "1",
        help=(
            "Run BMC per assertion by isolating each verif.assert in prepared MLIR "
            "(default: env BMC_ASSERTION_GRANULAR or off)."
        ),
    )
    parser.add_argument(
        "--assertion-granular-max",
        default=os.environ.get("BMC_ASSERTION_GRANULAR_MAX", "0"),
        help=(
            "Maximum assertions per case for --assertion-granular "
            "(0 means unlimited)."
        ),
    )
    parser.add_argument(
        "--assertion-shard-count",
        default=os.environ.get("BMC_ASSERTION_SHARD_COUNT", "1"),
        help=(
            "Optional number of deterministic assertion shards used when "
            "--assertion-granular is enabled "
            "(default: env BMC_ASSERTION_SHARD_COUNT or 1)."
        ),
    )
    parser.add_argument(
        "--assertion-shard-index",
        default=os.environ.get("BMC_ASSERTION_SHARD_INDEX", "0"),
        help=(
            "Optional deterministic assertion shard index in "
            "[0, assertion-shard-count) used when --assertion-granular "
            "is enabled (default: env BMC_ASSERTION_SHARD_INDEX or 0)."
        ),
    )
    parser.add_argument(
        "--cover-granular",
        action="store_true",
        default=os.environ.get("BMC_COVER_GRANULAR", "0") == "1",
        help=(
            "Run BMC per cover by isolating each verif.cover in prepared MLIR "
            "(default: env BMC_COVER_GRANULAR or off)."
        ),
    )
    parser.add_argument(
        "--cover-shard-count",
        default=os.environ.get("BMC_COVER_SHARD_COUNT", "1"),
        help=(
            "Optional number of deterministic cover shards used when "
            "--cover-granular is enabled "
            "(default: env BMC_COVER_SHARD_COUNT or 1)."
        ),
    )
    parser.add_argument(
        "--cover-shard-index",
        default=os.environ.get("BMC_COVER_SHARD_INDEX", "0"),
        help=(
            "Optional deterministic cover shard index in [0, cover-shard-count) "
            "used when --cover-granular is enabled "
            "(default: env BMC_COVER_SHARD_INDEX or 0)."
        ),
    )
    parser.add_argument(
        "--case-shard-count",
        default=os.environ.get("BMC_CASE_SHARD_COUNT", "1"),
        help=(
            "Optional number of deterministic case shards "
            "(default: env BMC_CASE_SHARD_COUNT or 1)."
        ),
    )
    parser.add_argument(
        "--case-shard-index",
        default=os.environ.get("BMC_CASE_SHARD_INDEX", "0"),
        help=(
            "Optional deterministic case shard index in [0, case-shard-count) "
            "(default: env BMC_CASE_SHARD_INDEX or 0)."
        ),
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
    verilog_single_unit_mode = os.environ.get(
        "BMC_VERILOG_SINGLE_UNIT_MODE", "auto"
    ).strip().lower()
    if verilog_single_unit_mode not in {"auto", "on", "off"}:
        print(
            (
                "invalid BMC_VERILOG_SINGLE_UNIT_MODE: "
                f"{verilog_single_unit_mode} (expected auto|on|off)"
            ),
            file=sys.stderr,
        )
        return 1
    verilog_xilinx_primitive_stub_mode = os.environ.get(
        "BMC_VERILOG_XILINX_PRIMITIVE_STUB_MODE", "auto"
    ).strip().lower()
    if verilog_xilinx_primitive_stub_mode not in {"auto", "on", "off"}:
        print(
            (
                "invalid BMC_VERILOG_XILINX_PRIMITIVE_STUB_MODE: "
                f"{verilog_xilinx_primitive_stub_mode} (expected auto|on|off)"
            ),
            file=sys.stderr,
        )
        return 1
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
    launch_retry_attempts = parse_nonnegative_int(
        os.environ.get("BMC_LAUNCH_RETRY_ATTEMPTS", "4"),
        "BMC_LAUNCH_RETRY_ATTEMPTS",
    )
    launch_retry_backoff_secs = parse_nonnegative_float(
        os.environ.get("BMC_LAUNCH_RETRY_BACKOFF_SECS", "0.2"),
        "BMC_LAUNCH_RETRY_BACKOFF_SECS",
    )
    launch_retryable_exit_codes = parse_exit_codes(
        os.environ.get("BMC_LAUNCH_RETRYABLE_EXIT_CODES", "126,127"),
        "BMC_LAUNCH_RETRYABLE_EXIT_CODES",
    )
    launch_copy_fallback_raw = os.environ.get("BMC_LAUNCH_COPY_FALLBACK", "0")
    if launch_copy_fallback_raw not in {"0", "1"}:
        print(
            (
                f"invalid BMC_LAUNCH_COPY_FALLBACK: {launch_copy_fallback_raw} "
                "(expected 0 or 1)"
            ),
            file=sys.stderr,
        )
        return 1
    launch_copy_fallback = launch_copy_fallback_raw == "1"
    bound = parse_nonnegative_int(args.bound, "--bound")
    if bound == 0:
        bound = 1
    ignore_asserts_until = parse_nonnegative_int(
        args.ignore_asserts_until, "--ignore-asserts-until"
    )
    assertion_granular_max = parse_nonnegative_int(
        args.assertion_granular_max, "--assertion-granular-max"
    )
    assertion_shard_count = parse_nonnegative_int(
        args.assertion_shard_count, "--assertion-shard-count"
    )
    if assertion_shard_count <= 0:
        print(
            (
                f"invalid --assertion-shard-count: {args.assertion_shard_count} "
                "(expected >= 1)"
            ),
            file=sys.stderr,
        )
        return 1
    assertion_shard_index = parse_nonnegative_int(
        args.assertion_shard_index, "--assertion-shard-index"
    )
    if assertion_shard_index >= assertion_shard_count:
        print(
            (
                f"invalid --assertion-shard-index: {assertion_shard_index} "
                f"(expected < {assertion_shard_count})"
            ),
            file=sys.stderr,
        )
        return 1
    cover_shard_count = parse_nonnegative_int(
        args.cover_shard_count, "--cover-shard-count"
    )
    if cover_shard_count <= 0:
        print(
            (
                f"invalid --cover-shard-count: {args.cover_shard_count} "
                "(expected >= 1)"
            ),
            file=sys.stderr,
        )
        return 1
    cover_shard_index = parse_nonnegative_int(
        args.cover_shard_index, "--cover-shard-index"
    )
    if cover_shard_index >= cover_shard_count:
        print(
            (
                f"invalid --cover-shard-index: {cover_shard_index} "
                f"(expected < {cover_shard_count})"
            ),
            file=sys.stderr,
        )
        return 1
    case_shard_count = parse_nonnegative_int(
        args.case_shard_count, "--case-shard-count"
    )
    if case_shard_count <= 0:
        print(
            (
                f"invalid --case-shard-count: {args.case_shard_count} "
                "(expected >= 1)"
            ),
            file=sys.stderr,
        )
        return 1
    case_shard_index = parse_nonnegative_int(
        args.case_shard_index, "--case-shard-index"
    )
    if case_shard_index >= case_shard_count:
        print(
            (
                f"invalid --case-shard-index: {case_shard_index} "
                f"(expected < {case_shard_count})"
            ),
            file=sys.stderr,
        )
        return 1

    def case_shard_key(case: CaseSpec) -> tuple[str, ...]:
        return (
            case.case_id,
            case.top_module,
            ";".join(case.source_files),
            ";".join(case.include_dirs),
            case.case_path,
            str(case.timeout_secs if case.timeout_secs is not None else ""),
            case.backend_mode,
            str(case.bmc_bound if case.bmc_bound is not None else ""),
            str(
                case.ignore_asserts_until
                if case.ignore_asserts_until is not None
                else ""
            ),
            case.assume_known_inputs_mode,
            case.allow_multi_clock_mode,
            shlex.join(case.bmc_extra_args),
            case.contract_source,
            ";".join(case.verilog_defines),
        )

    all_case_keys = sorted({case_shard_key(case) for case in cases})
    shard_case_keys = {
        key
        for idx, key in enumerate(all_case_keys)
        if idx % case_shard_count == case_shard_index
    }
    cases = [case for case in cases if case_shard_key(case) in shard_case_keys]
    print(
        "pairwise BMC shard selection: "
        f"shard={case_shard_index}/{case_shard_count} "
        f"selected_cases={len(shard_case_keys)} total_cases={len(all_case_keys)}",
        flush=True,
    )
    print(
        "pairwise BMC objective sharding: "
        f"assertion={assertion_shard_index}/{assertion_shard_count} "
        f"cover={cover_shard_index}/{cover_shard_count}",
        flush=True,
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
    assertion_result_rows: list[tuple[str, ...]] = []
    cover_result_rows: list[tuple[str, ...]] = []
    launch_event_rows: list[tuple[str, ...]] = []
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
    skipped = 0
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
            if case.verilog_defines:
                contract_fields.append(";".join(case.verilog_defines))
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
            def build_verilog_cmd(
                use_single_unit: bool, extra_sources: list[str] | None = None
            ) -> list[str]:
                cmd = [
                    circt_verilog,
                    "--ir-hw",
                    "-o",
                    str(out_mlir),
                    "--no-uvm-auto-include",
                    f"--top={case.top_module}",
                ]
                if use_single_unit:
                    cmd.append("--single-unit")
                for include_dir in include_dirs:
                    cmd += ["-I", include_dir]
                for define_token in case.verilog_defines:
                    cmd.append(f"-D{define_token}")
                cmd += circt_verilog_args
                cmd += case.source_files
                if extra_sources:
                    cmd += extra_sources
                return cmd

            use_single_unit = verilog_single_unit_mode != "off"
            xilinx_primitive_stub_path = (
                case_dir / "circt-verilog.xilinx-primitives.sv"
            )
            verilog_extra_sources: list[str] = []
            if verilog_xilinx_primitive_stub_mode == "on":
                write_xilinx_primitive_stub_file(xilinx_primitive_stub_path)
                verilog_extra_sources.append(str(xilinx_primitive_stub_path))
            verilog_cmd = build_verilog_cmd(use_single_unit, verilog_extra_sources)

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
                    launch_retry_attempts,
                    launch_retry_backoff_secs,
                    launch_retryable_exit_codes,
                    launch_copy_fallback,
                    launch_event_rows,
                    case.case_id,
                    case_path,
                    stage,
                )
                if (
                    verilog_result.returncode != 0
                    and verilog_single_unit_mode == "auto"
                    and use_single_unit
                ):
                    verilog_log_text = verilog_log_path.read_text()
                    if is_single_unit_retryable_preprocessor_failure(
                        verilog_log_text
                    ):
                        single_unit_log_path = (
                            case_dir / "circt-verilog.single-unit.log"
                        )
                        shutil.copy2(verilog_log_path, single_unit_log_path)
                        if launch_event_rows is not None:
                            launch_event_rows.append(
                                (
                                    "RETRY",
                                    case.case_id,
                                    case_path,
                                    stage,
                                    circt_verilog,
                                    "single_unit_preprocessor_failure",
                                    "1",
                                    "0.000",
                                    str(verilog_result.returncode),
                                    "",
                                )
                            )
                        use_single_unit = False
                        verilog_cmd = build_verilog_cmd(
                            use_single_unit, verilog_extra_sources
                        )
                        verilog_result = run_and_log(
                            verilog_cmd,
                            verilog_log_path,
                            None,
                            case_timeout_secs,
                            etxtbsy_retries,
                            etxtbsy_backoff_secs,
                            launch_retry_attempts,
                            launch_retry_backoff_secs,
                            launch_retryable_exit_codes,
                            launch_copy_fallback,
                            launch_event_rows,
                            case.case_id,
                            case_path,
                            stage,
                        )
                if (
                    verilog_result.returncode != 0
                    and verilog_xilinx_primitive_stub_mode == "auto"
                    and not verilog_extra_sources
                ):
                    verilog_log_text = verilog_log_path.read_text()
                    if is_xilinx_primitive_stub_retryable_failure(verilog_log_text):
                        unknown_module_log_path = (
                            case_dir / "circt-verilog.unknown-module.log"
                        )
                        shutil.copy2(verilog_log_path, unknown_module_log_path)
                        write_xilinx_primitive_stub_file(xilinx_primitive_stub_path)
                        verilog_extra_sources = [str(xilinx_primitive_stub_path)]
                        if launch_event_rows is not None:
                            launch_event_rows.append(
                                (
                                    "RETRY",
                                    case.case_id,
                                    case_path,
                                    stage,
                                    circt_verilog,
                                    "unknown_module_xilinx_primitives",
                                    "1",
                                    "0.000",
                                    str(verilog_result.returncode),
                                    "",
                                )
                            )
                        verilog_cmd = build_verilog_cmd(
                            use_single_unit, verilog_extra_sources
                        )
                        verilog_result = run_and_log(
                            verilog_cmd,
                            verilog_log_path,
                            None,
                            case_timeout_secs,
                            etxtbsy_retries,
                            etxtbsy_backoff_secs,
                            launch_retry_attempts,
                            launch_retry_backoff_secs,
                            launch_retryable_exit_codes,
                            launch_copy_fallback,
                            launch_event_rows,
                            case.case_id,
                            case_path,
                            stage,
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
                        launch_retry_attempts,
                        launch_retry_backoff_secs,
                        launch_retryable_exit_codes,
                        launch_copy_fallback,
                        launch_event_rows,
                        case.case_id,
                        case_path,
                        stage,
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

                bmc_input_mlir = (
                    prepped_mlir if bmc_prepare_core_with_circt_opt else out_mlir
                )
                mlir_lines: list[str] | None = None
                if (args.assertion_granular or args.cover_granular) and not case_smoke_only:
                    mlir_lines = bmc_input_mlir.read_text(encoding="utf-8").splitlines()

                if args.cover_granular and not case_smoke_only:
                    cover_sites = collect_candidate_cover_sites(mlir_lines or [])
                    selected_cover_sites = [
                        site
                        for site in cover_sites
                        if site.ordinal % cover_shard_count == cover_shard_index
                    ]
                    if cover_sites:
                        assertion_sites_for_cover = collect_candidate_assertion_sites(
                            mlir_lines or []
                        )
                        for site in selected_cover_sites:
                            cover_id = f"c{site.ordinal:04d}"
                            cover_label = f"line_{site.line_index + 1}"
                            cover_mlir = case_dir / f"pairwise_bmc.cover-{site.ordinal}.mlir"
                            cover_mlir.write_text(
                                build_isolated_cover_mlir(
                                    mlir_lines or [],
                                    assertion_sites_for_cover,
                                    cover_sites,
                                    site.ordinal,
                                ),
                                encoding="utf-8",
                            )
                            cover_log = case_dir / f"circt-bmc.cover-{site.ordinal}.log"
                            cover_out = case_dir / f"circt-bmc.cover-{site.ordinal}.out"
                            cover_cmd = list(bmc_cmd)
                            cover_cmd[1] = str(cover_mlir)
                            stage = f"bmc.cover.{site.ordinal}"
                            try:
                                cover_run = run_and_log(
                                    cover_cmd,
                                    cover_log,
                                    cover_out,
                                    case_timeout_secs,
                                    etxtbsy_retries,
                                    etxtbsy_backoff_secs,
                                    launch_retry_attempts,
                                    launch_retry_backoff_secs,
                                    launch_retryable_exit_codes,
                                    launch_copy_fallback,
                                    launch_event_rows,
                                    case.case_id,
                                    case_path,
                                    stage,
                                )
                                combined = (
                                    cover_log.read_text() + "\n" + cover_out.read_text()
                                )
                                bmc_tag = parse_bmc_result(combined)
                                if bmc_tag == "SAT":
                                    cover_result_rows.append(
                                        (
                                            "COVERED",
                                            case.case_id,
                                            case_path,
                                            cover_id,
                                            cover_label,
                                            "SAT",
                                            "sat",
                                        )
                                    )
                                elif bmc_tag == "UNSAT":
                                    cover_result_rows.append(
                                        (
                                            "UNREACHABLE",
                                            case.case_id,
                                            case_path,
                                            cover_id,
                                            cover_label,
                                            "UNSAT",
                                            "unsat",
                                        )
                                    )
                                elif bmc_tag == "UNKNOWN":
                                    cover_result_rows.append(
                                        (
                                            "UNKNOWN",
                                            case.case_id,
                                            case_path,
                                            cover_id,
                                            cover_label,
                                            "UNKNOWN",
                                            "unknown",
                                        )
                                    )
                                elif cover_run.returncode != 0:
                                    reason = normalize_error_reason(combined)
                                    cover_result_rows.append(
                                        (
                                            "ERROR",
                                            case.case_id,
                                            case_path,
                                            cover_id,
                                            cover_label,
                                            "CIRCT_BMC_ERROR",
                                            reason,
                                        )
                                    )
                                else:
                                    reason = normalize_error_reason(combined)
                                    cover_result_rows.append(
                                        (
                                            "ERROR",
                                            case.case_id,
                                            case_path,
                                            cover_id,
                                            cover_label,
                                            "CIRCT_BMC_ERROR",
                                            reason,
                                        )
                                    )
                            except subprocess.TimeoutExpired:
                                cover_result_rows.append(
                                    (
                                        "TIMEOUT",
                                        case.case_id,
                                        case_path,
                                        cover_id,
                                        cover_label,
                                        "BMC_TIMEOUT",
                                        "solver_command_timeout",
                                    )
                                )
                            except Exception as exc:
                                append_exception_log(cover_log, stage, exc)
                                if isinstance(exc, TextFileBusyRetryExhausted):
                                    reason = (
                                        "runner_command_exception_bmc_"
                                        "runner_command_etxtbsy_retry_exhausted"
                                    )
                                else:
                                    reason_body = normalize_error_reason(
                                        f"{type(exc).__name__}: {exc}"
                                    )
                                    reason = (
                                        "runner_command_exception"
                                        if reason_body == "no_diag"
                                        else f"runner_command_exception_bmc_{reason_body}"
                                    )
                                cover_result_rows.append(
                                    (
                                        "ERROR",
                                        case.case_id,
                                        case_path,
                                        cover_id,
                                        cover_label,
                                        "CIRCT_BMC_ERROR",
                                        reason,
                                    )
                                )

                if args.assertion_granular and not case_smoke_only:
                    assertion_sites = collect_candidate_assertion_sites(mlir_lines or [])
                    selected_assertion_sites = [
                        site
                        for site in assertion_sites
                        if site.ordinal % assertion_shard_count == assertion_shard_index
                    ]
                    if (
                        assertion_granular_max > 0
                        and len(selected_assertion_sites) > assertion_granular_max
                    ):
                        rows.append(
                            (
                                "ERROR",
                                case.case_id,
                                case_path,
                                suite_name,
                                mode_label,
                                "CIRCT_BMC_ERROR",
                                "assertion_granular_limit_exceeded",
                            )
                        )
                        errored += 1
                        print(
                            (
                                f"{case.case_id:32} ERROR "
                                "(assertion_granular_limit_exceeded)"
                            ),
                            flush=True,
                        )
                        continue
                    if assertion_sites:
                        if not selected_assertion_sites:
                            rows.append(
                                (
                                    "SKIP",
                                    case.case_id,
                                    case_path,
                                    suite_name,
                                    mode_label,
                                    "BMC_NOT_RUN",
                                    "assertion_shard_empty",
                                )
                            )
                            skipped += 1
                            print(
                                (
                                    f"{case.case_id:32} SKIP "
                                    "(assertion_shard_empty)"
                                ),
                                flush=True,
                            )
                            continue
                        assertion_statuses: list[tuple[str, str]] = []
                        for site in selected_assertion_sites:
                            assertion_id = f"a{site.ordinal:04d}"
                            assertion_label = f"line_{site.line_index + 1}"
                            assertion_mlir = (
                                case_dir / f"pairwise_bmc.assertion-{site.ordinal}.mlir"
                            )
                            assertion_mlir.write_text(
                                build_isolated_assertion_mlir(
                                    mlir_lines, assertion_sites, site.ordinal
                                ),
                                encoding="utf-8",
                            )
                            assertion_log = (
                                case_dir / f"circt-bmc.assertion-{site.ordinal}.log"
                            )
                            assertion_out = (
                                case_dir / f"circt-bmc.assertion-{site.ordinal}.out"
                            )
                            assertion_cmd = list(bmc_cmd)
                            assertion_cmd[1] = str(assertion_mlir)
                            stage = f"bmc.assertion.{site.ordinal}"
                            try:
                                assertion_run = run_and_log(
                                    assertion_cmd,
                                    assertion_log,
                                    assertion_out,
                                    case_timeout_secs,
                                    etxtbsy_retries,
                                    etxtbsy_backoff_secs,
                                    launch_retry_attempts,
                                    launch_retry_backoff_secs,
                                    launch_retryable_exit_codes,
                                    launch_copy_fallback,
                                    launch_event_rows,
                                    case.case_id,
                                    case_path,
                                    stage,
                                )
                                combined = (
                                    assertion_log.read_text() + "\n" + assertion_out.read_text()
                                )
                                assertion_tag = parse_assertion_status(combined)
                                bmc_tag = parse_bmc_result(combined)
                                if assertion_tag == "VACUOUS":
                                    assertion_result_rows.append(
                                        (
                                            "VACUOUS",
                                            case.case_id,
                                            case_path,
                                            assertion_id,
                                            assertion_label,
                                            "UNSAT",
                                            "vacuous",
                                        )
                                    )
                                    assertion_statuses.append(("VACUOUS", "vacuous"))
                                elif assertion_tag == "PROVEN":
                                    assertion_result_rows.append(
                                        (
                                            "PROVEN",
                                            case.case_id,
                                            case_path,
                                            assertion_id,
                                            assertion_label,
                                            "UNSAT",
                                            "unsat",
                                        )
                                    )
                                    assertion_statuses.append(("PROVEN", "unsat"))
                                elif assertion_tag == "FAILING":
                                    assertion_result_rows.append(
                                        (
                                            "FAILING",
                                            case.case_id,
                                            case_path,
                                            assertion_id,
                                            assertion_label,
                                            "SAT",
                                            "sat",
                                        )
                                    )
                                    assertion_statuses.append(("FAILING", "sat"))
                                elif assertion_tag == "UNKNOWN":
                                    assertion_result_rows.append(
                                        (
                                            "UNKNOWN",
                                            case.case_id,
                                            case_path,
                                            assertion_id,
                                            assertion_label,
                                            "UNKNOWN",
                                            "unknown",
                                        )
                                    )
                                    assertion_statuses.append(("UNKNOWN", "unknown"))
                                elif bmc_tag == "UNSAT":
                                    assertion_result_rows.append(
                                        (
                                            "PROVEN",
                                            case.case_id,
                                            case_path,
                                            assertion_id,
                                            assertion_label,
                                            "UNSAT",
                                            "unsat",
                                        )
                                    )
                                    assertion_statuses.append(("PROVEN", "unsat"))
                                elif bmc_tag == "SAT":
                                    assertion_result_rows.append(
                                        (
                                            "FAILING",
                                            case.case_id,
                                            case_path,
                                            assertion_id,
                                            assertion_label,
                                            "SAT",
                                            "sat",
                                        )
                                    )
                                    assertion_statuses.append(("FAILING", "sat"))
                                elif bmc_tag == "UNKNOWN":
                                    assertion_result_rows.append(
                                        (
                                            "UNKNOWN",
                                            case.case_id,
                                            case_path,
                                            assertion_id,
                                            assertion_label,
                                            "UNKNOWN",
                                            "unknown",
                                        )
                                    )
                                    assertion_statuses.append(("UNKNOWN", "unknown"))
                                elif assertion_run.returncode != 0:
                                    reason = normalize_error_reason(combined)
                                    assertion_result_rows.append(
                                        (
                                            "ERROR",
                                            case.case_id,
                                            case_path,
                                            assertion_id,
                                            assertion_label,
                                            "CIRCT_BMC_ERROR",
                                            reason,
                                        )
                                    )
                                    assertion_statuses.append(("ERROR", reason))
                                else:
                                    reason = normalize_error_reason(combined)
                                    assertion_result_rows.append(
                                        (
                                            "ERROR",
                                            case.case_id,
                                            case_path,
                                            assertion_id,
                                            assertion_label,
                                            "CIRCT_BMC_ERROR",
                                            reason,
                                        )
                                    )
                                    assertion_statuses.append(("ERROR", reason))
                            except subprocess.TimeoutExpired:
                                assertion_result_rows.append(
                                    (
                                        "TIMEOUT",
                                        case.case_id,
                                        case_path,
                                        assertion_id,
                                        assertion_label,
                                        "BMC_TIMEOUT",
                                        "solver_command_timeout",
                                    )
                                )
                                assertion_statuses.append(
                                    ("TIMEOUT", "solver_command_timeout")
                                )
                            except Exception as exc:
                                append_exception_log(assertion_log, stage, exc)
                                if isinstance(exc, TextFileBusyRetryExhausted):
                                    reason = (
                                        "runner_command_exception_bmc_"
                                        "runner_command_etxtbsy_retry_exhausted"
                                    )
                                else:
                                    reason_body = normalize_error_reason(
                                        f"{type(exc).__name__}: {exc}"
                                    )
                                    reason = (
                                        "runner_command_exception"
                                        if reason_body == "no_diag"
                                        else f"runner_command_exception_bmc_{reason_body}"
                                    )
                                assertion_result_rows.append(
                                    (
                                        "ERROR",
                                        case.case_id,
                                        case_path,
                                        assertion_id,
                                        assertion_label,
                                        "CIRCT_BMC_ERROR",
                                        reason,
                                    )
                                )
                                assertion_statuses.append(("ERROR", reason))

                        status_set = {status for status, _ in assertion_statuses}
                        if "ERROR" in status_set:
                            reason = next(
                                reason
                                for status, reason in assertion_statuses
                                if status == "ERROR"
                            )
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
                            print(
                                f"{case.case_id:32} ERROR (CIRCT_BMC_ERROR)",
                                flush=True,
                            )
                        elif "TIMEOUT" in status_set:
                            rows.append(
                                (
                                    "TIMEOUT",
                                    case.case_id,
                                    case_path,
                                    suite_name,
                                    mode_label,
                                    "BMC_TIMEOUT",
                                    "solver_command_timeout",
                                )
                            )
                            timeout += 1
                            print(
                                f"{case.case_id:32} TIMEOUT (BMC_TIMEOUT)",
                                flush=True,
                            )
                        elif "FAILING" in status_set:
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
                        elif "UNKNOWN" in status_set:
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
                        continue

                stage = "bmc"
                bmc_result = run_and_log(
                    bmc_cmd,
                    bmc_log_path,
                    bmc_out_path,
                    case_timeout_secs,
                    etxtbsy_retries,
                    etxtbsy_backoff_secs,
                    launch_retry_attempts,
                    launch_retry_backoff_secs,
                    launch_retryable_exit_codes,
                    launch_copy_fallback,
                    launch_event_rows,
                    case.case_id,
                    case_path,
                    stage,
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

    if args.assertion_results_file:
        assertion_results_path = Path(args.assertion_results_file)
        assertion_results_path.parent.mkdir(parents=True, exist_ok=True)
        with assertion_results_path.open("w", encoding="utf-8") as handle:
            for row in sorted(
                assertion_result_rows, key=lambda item: (item[1], item[3], item[0])
            ):
                handle.write("\t".join(row) + "\n")

    if args.cover_results_file:
        cover_results_path = Path(args.cover_results_file)
        cover_results_path.parent.mkdir(parents=True, exist_ok=True)
        with cover_results_path.open("w", encoding="utf-8") as handle:
            for row in sorted(
                cover_result_rows, key=lambda item: (item[1], item[3], item[0])
            ):
                handle.write("\t".join(row) + "\n")

    if args.launch_events_file:
        launch_events_path = Path(args.launch_events_file)
        launch_events_path.parent.mkdir(parents=True, exist_ok=True)
        with launch_events_path.open("w", encoding="utf-8") as handle:
            for row in sorted(
                launch_event_rows, key=lambda item: (item[1], item[3], item[0], item[6])
            ):
                handle.write("\t".join(row) + "\n")

    print(
        f"{suite_name} BMC dropped-syntax summary: "
        f"drop_remark_cases={len(set(drop_remark_case_rows))} "
        f"pattern='{drop_remark_pattern}'",
        flush=True,
    )
    if args.assertion_granular:
        print(
            f"{suite_name} BMC assertion-granular summary: "
            f"assertions={len(assertion_result_rows)}",
            flush=True,
        )
    if args.cover_granular:
        print(
            f"{suite_name} BMC cover-granular summary: "
            f"covers={len(cover_result_rows)}",
            flush=True,
        )
    if launch_event_rows:
        retry_events = sum(1 for row in launch_event_rows if row and row[0] == "RETRY")
        fallback_events = sum(
            1 for row in launch_event_rows if row and row[0] == "FALLBACK"
        )
        print(
            f"{suite_name} BMC launch-events summary: "
            f"events={len(launch_event_rows)} retry={retry_events} "
            f"fallback={fallback_events}",
            flush=True,
        )
    print(
        f"{suite_name} BMC summary: total={total} pass={passed} fail={failed} "
        f"xfail=0 xpass=0 error={errored} skip={skipped} unknown={unknown} timeout={timeout}",
        flush=True,
    )

    if failed or errored or unknown or timeout:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

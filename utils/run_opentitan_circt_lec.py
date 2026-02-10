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
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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


def parse_lec_result(text: str) -> str | None:
    match = re.search(r"LEC_RESULT=(EQ|NEQ|UNKNOWN)", text)
    if not match:
        return None
    return match.group(1)

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


def run_and_log(
    cmd: list[str], log_path: Path, out_path: Path | None = None
) -> str:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    write_log(log_path, result.stdout, result.stderr)
    if out_path is not None:
        out_path.write_text(result.stdout)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    return result.stdout


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
        "--xprop-summary-file",
        default=os.environ.get("OUT_XPROP_SUMMARY", ""),
        help=(
            "Optional TSV output path for XPROP diagnostic rows "
            "(implementation, mode, diag, result, counters, log_dir, "
            "assume_known_result)."
        ),
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

    circt_verilog = os.environ.get("CIRCT_VERILOG", "build/bin/circt-verilog")
    circt_verilog_args = shlex.split(os.environ.get("CIRCT_VERILOG_ARGS", ""))
    circt_opt = os.environ.get("CIRCT_OPT", "build/bin/circt-opt")
    circt_opt_args = shlex.split(os.environ.get("CIRCT_OPT_ARGS", ""))
    circt_lec = os.environ.get("CIRCT_LEC", "build/bin/circt-lec")
    circt_lec_args = shlex.split(os.environ.get("CIRCT_LEC_ARGS", ""))
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
    lec_smoke_only = os.environ.get("LEC_SMOKE_ONLY", "0") == "1"
    lec_run_smtlib = os.environ.get("LEC_RUN_SMTLIB", "1") == "1"
    lec_mode_label = os.environ.get("LEC_MODE_LABEL", "LEC").strip() or "LEC"
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
    case_rows: list[tuple[str, str, str, str, str]] = []
    xprop_rows: list[tuple[str, str, str, str, str, str, str, str]] = []
    try:
        print(
            f"Running LEC on {len(impl_list)} AES S-Box implementation(s)...",
            flush=True,
        )
        for impl in sorted(impl_list):
            impl_dir = workdir / impl
            impl_dir.mkdir(parents=True, exist_ok=True)
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

            result: str | None = None
            diag: str | None = None
            assume_known_result: str | None = None
            summary_counts: dict[str, int] = {}
            try:
                run_and_log(verilog_cmd, impl_dir / "circt-verilog.log")
                run_and_log(opt_cmd, impl_dir / "circt-opt.log")
                lec_stdout = run_and_log(
                    lec_cmd,
                    impl_dir / "circt-lec.log",
                    out_path=impl_dir / "circt-lec.out",
                )
                if not lec_smoke_only:
                    lec_log_text = (impl_dir / "circt-lec.log").read_text()
                    combined = lec_log_text + "\n" + lec_stdout
                    result = parse_lec_result(combined)
                    diag = parse_lec_diag(combined)
                    assume_known_result = parse_lec_diag_assume_known_result(combined)
                    if diag == "XPROP_ONLY":
                        summary_counts = parse_xprop_summary_counts(combined)
                    if result in ("NEQ", "UNKNOWN"):
                        if diag == "XPROP_ONLY" and lec_accept_xprop_only:
                            print(
                                f"{impl:24} XPROP_ONLY (accepted)",
                                flush=True,
                            )
                            case_rows.append(
                                ("XFAIL", impl, f"{impl_dir}#XPROP_ONLY", "opentitan", lec_mode_label)
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
                print(f"{impl:24} OK", flush=True)
                case_rows.append(("PASS", impl, str(impl_dir), "opentitan", lec_mode_label))
            except subprocess.CalledProcessError:
                failures += 1
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
                print(f"{impl:24} FAIL{extra} (logs in {impl_dir})", flush=True)
                detail = str(impl_dir)
                if diag:
                    detail = f"{detail}#{diag}"
                case_rows.append(("FAIL", impl, detail, "opentitan", lec_mode_label))
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
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w") as handle:
            for row in sorted(case_rows, key=lambda item: (item[1], item[0], item[2])):
                handle.write("\t".join(row) + "\n")
    if args.xprop_summary_file:
        xprop_summary_path = Path(args.xprop_summary_file)
        xprop_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with xprop_summary_path.open("w") as handle:
            for row in sorted(xprop_rows, key=lambda item: (item[1], item[2], item[0], item[6])):
                handle.write("\t".join(row) + "\n")

    if failures:
        print(f"LEC failures: {failures}", file=sys.stderr)
        return 1
    print("LEC completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

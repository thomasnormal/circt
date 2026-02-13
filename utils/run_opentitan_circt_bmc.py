#!/usr/bin/env python3
# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Run OpenTitan AES S-Box BMC checks with circt-bmc.

This script is intentionally OpenTitan-specific only for case discovery and
miter generation. Actual BMC execution is delegated to the generic
`run_pairwise_circt_bmc.py` runner so the backend remains reusable across
projects.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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


def parse_optional_nonnegative_int(raw: str, name: str, line_no: int) -> int | None:
    token = raw.strip()
    if not token:
        return None
    try:
        value = int(token)
    except ValueError:
        print(
            f"invalid case-policy row {line_no}: {name} must be non-negative integer, got '{raw}'",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if value < 0:
        print(
            f"invalid case-policy row {line_no}: {name} must be non-negative integer, got '{raw}'",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value


def parse_case_backend_mode(raw: str, line_no: int) -> str:
    token = raw.strip().lower()
    if not token:
        return "default"
    if token in {"default", "jit", "smtlib", "smoke"}:
        return token
    print(
        (
            f"invalid case-policy row {line_no}: backend_mode must be one of "
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
            f"invalid case-policy row {line_no}: {name} must be one of "
            f"default|on|off (or 1|0|true|false|yes|no), got '{raw}'"
        ),
        file=sys.stderr,
    )
    raise SystemExit(1)


def compile_case_policy_pattern(
    selector_raw: str, line_no: int, selector_kind: str
) -> re.Pattern[str]:
    pattern_text = selector_raw.strip()
    if not pattern_text:
        print(
            f"invalid case-policy row {line_no}: empty {selector_kind}: selector",
            file=sys.stderr,
        )
        raise SystemExit(1)

    regex_text = pattern_text
    if selector_kind == "glob":
        regex_text = fnmatch.translate(pattern_text)

    try:
        return re.compile(regex_text)
    except re.error as exc:
        print(
            (
                f"invalid case-policy row {line_no}: invalid {selector_kind}: selector "
                f"'{pattern_text}': {exc}"
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)


def parse_case_bmc_extra_args(raw: str, line_no: int) -> str:
    token = raw.strip()
    if not token:
        return ""
    try:
        args = shlex.split(token)
    except ValueError as exc:
        print(
            f"invalid case-policy row {line_no}: bmc_extra_args parse error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return shlex.join(args)


def load_case_policy(
    policy_file: Path,
) -> tuple[
    dict[str, dict[str, object]],
    list[tuple[str, re.Pattern[str], dict[str, object]]],
    dict[str, object] | None,
]:
    policy_by_impl: dict[str, dict[str, object]] = {}
    pattern_rules: list[tuple[str, re.Pattern[str], dict[str, object]]] = []
    pattern_rule_keys: set[str] = set()
    default_policy: dict[str, object] | None = None
    with policy_file.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) > 8:
                print(
                    (
                        f"invalid case-policy row {line_no}: expected at most 8 tab "
                        "columns (impl_selector, timeout_secs, backend_mode, bmc_bound, "
                        "ignore_asserts_until, assume_known_inputs, allow_multi_clock, bmc_extra_args)"
                    ),
                    file=sys.stderr,
                )
                raise SystemExit(1)
            while len(parts) < 8:
                parts.append("")

            impl_key = parts[0].strip()
            if not impl_key:
                print(
                    f"invalid case-policy row {line_no}: empty impl selector",
                    file=sys.stderr,
                )
                raise SystemExit(1)

            entry: dict[str, object] = {
                "timeout_secs": parse_optional_nonnegative_int(
                    parts[1], "timeout_secs", line_no
                ),
                "backend_mode": parse_case_backend_mode(parts[2], line_no),
                "bmc_bound": parse_optional_nonnegative_int(
                    parts[3], "bmc_bound", line_no
                ),
                "ignore_asserts_until": parse_optional_nonnegative_int(
                    parts[4], "ignore_asserts_until", line_no
                ),
                "assume_known_inputs": parse_case_toggle_mode(
                    parts[5], line_no, "assume_known_inputs"
                ),
                "allow_multi_clock": parse_case_toggle_mode(
                    parts[6], line_no, "allow_multi_clock"
                ),
                "bmc_extra_args": parse_case_bmc_extra_args(parts[7], line_no),
            }

            if impl_key == "*":
                if default_policy is not None:
                    print(
                        "invalid case-policy: multiple '*' default rows",
                        file=sys.stderr,
                    )
                    raise SystemExit(1)
                default_policy = entry
                continue

            if impl_key.startswith("re:"):
                if impl_key in pattern_rule_keys:
                    print(
                        f"invalid case-policy: duplicate selector '{impl_key}'",
                        file=sys.stderr,
                    )
                    raise SystemExit(1)
                pattern_rule_keys.add(impl_key)
                pattern_rules.append(
                    (
                        impl_key,
                        compile_case_policy_pattern(
                            impl_key[len("re:") :], line_no, "re"
                        ),
                        entry,
                    )
                )
                continue

            if impl_key.startswith("glob:"):
                if impl_key in pattern_rule_keys:
                    print(
                        f"invalid case-policy: duplicate selector '{impl_key}'",
                        file=sys.stderr,
                    )
                    raise SystemExit(1)
                pattern_rule_keys.add(impl_key)
                pattern_rules.append(
                    (
                        impl_key,
                        compile_case_policy_pattern(
                            impl_key[len("glob:") :], line_no, "glob"
                        ),
                        entry,
                    )
                )
                continue

            if impl_key in policy_by_impl:
                print(
                    f"invalid case-policy: duplicate implementation key '{impl_key}'",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            policy_by_impl[impl_key] = entry

    return policy_by_impl, pattern_rules, default_policy


def resolve_case_policy_for_impl(
    impl: str,
    policy_by_impl: dict[str, dict[str, object]],
    pattern_rules: list[tuple[str, re.Pattern[str], dict[str, object]]],
    default_policy: dict[str, object] | None,
) -> dict[str, object] | None:
    exact_policy = policy_by_impl.get(impl)
    if exact_policy is not None:
        return exact_policy

    matched_patterns: list[tuple[str, dict[str, object]]] = []
    for selector_key, selector_re, entry in pattern_rules:
        if selector_re.search(impl):
            matched_patterns.append((selector_key, entry))

    if len(matched_patterns) > 1:
        matched_keys = ", ".join(selector for selector, _ in matched_patterns)
        print(
            (
                f"ambiguous case-policy for implementation '{impl}': multiple "
                f"pattern selectors matched ({matched_keys})"
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)

    if len(matched_patterns) == 1:
        return matched_patterns[0][1]

    return default_policy


def replace_text(src: Path, dst: Path, replacements: list[tuple[str, str]]) -> None:
    data = src.read_text()
    for old, new in replacements:
        data = data.replace(old, new)
    dst.write_text(data)


def pick_prim(prim_xilinx_path: Path, name: str, fallback: str) -> str:
    preferred = prim_xilinx_path / name
    if preferred.exists():
        return str(preferred)
    return str(prim_xilinx_path / fallback)


def write_valid_op_wrapper(out_path: Path, wrapper_name: str, inner_name: str) -> None:
    out_path.write_text(
        "\n".join(
            [
                f"module {wrapper_name} (",
                "  input  aes_pkg::ciph_op_e op_i,",
                "  input  logic [7:0]        data_i,",
                "  output logic [7:0]        data_o",
                ");",
                "  always_comb begin",
                "    assume (op_i == aes_pkg::CIPH_FWD || op_i == aes_pkg::CIPH_INV);",
                "  end",
                f"  {inner_name} u_dut (",
                "    .op_i   ( op_i ),",
                "    .data_i ( data_i ),",
                "    .data_o ( data_o )",
                "  );",
                "endmodule",
                "",
            ]
        )
    )


def sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not sanitized:
        return "anon"
    if not re.match(r"[A-Za-z_]", sanitized[0]):
        sanitized = f"m_{sanitized}"
    return sanitized


def write_bmc_miter(
    out_path: Path, miter_name: str, ref_module: str, dut_module: str
) -> None:
    out_path.write_text(
        "\n".join(
            [
                f"module {miter_name} (",
                "  input  aes_pkg::ciph_op_e op_i,",
                "  input  logic [7:0]        data_i",
                ");",
                "  logic [7:0] ref_data_o;",
                "  logic [7:0] dut_data_o;",
                f"  {ref_module} u_ref (",
                "    .op_i   ( op_i ),",
                "    .data_i ( data_i ),",
                "    .data_o ( ref_data_o )",
                "  );",
                f"  {dut_module} u_dut (",
                "    .op_i   ( op_i ),",
                "    .data_i ( data_i ),",
                "    .data_o ( dut_data_o )",
                "  );",
                "  always_comb begin",
                "    assume (op_i == aes_pkg::CIPH_FWD || op_i == aes_pkg::CIPH_INV);",
                "    assert (ref_data_o == dut_data_o);",
                "  end",
                "endmodule",
                "",
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OpenTitan AES S-Box BMC checks with circt-bmc."
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
        "--case-policy-file",
        default="",
        help=(
            "Optional TSV policy manifest with per-implementation selectors "
            "(impl, re:<regex>, glob:<glob>, or * default)."
        ),
    )
    parser.add_argument(
        "--allow-invalid-op",
        action="store_true",
        help="Do not constrain invalid op_i values to supported enum values.",
    )
    parser.add_argument(
        "--results-file",
        default=os.environ.get("OUT", ""),
        help="Optional TSV output path for per-implementation case rows.",
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
        help="Optional TSV output path for timeout reason rows (case, path, reason).",
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

    policy_by_impl: dict[str, dict[str, object]] = {}
    policy_pattern_rules: list[tuple[str, re.Pattern[str], dict[str, object]]] = []
    default_policy: dict[str, object] | None = None
    if args.case_policy_file:
        policy_path = Path(args.case_policy_file).resolve()
        if not policy_path.is_file():
            print(f"case-policy file not found: {policy_path}", file=sys.stderr)
            return 1
        policy_by_impl, policy_pattern_rules, default_policy = load_case_policy(policy_path)

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

    mode_label = os.environ.get("BMC_MODE_LABEL", "BMC").strip() or "BMC"
    bound = parse_nonnegative_int(os.environ.get("BOUND", "1"), "BOUND")
    if bound == 0:
        bound = 1
    ignore_asserts_until = parse_nonnegative_int(
        os.environ.get("IGNORE_ASSERTS_UNTIL", "0"), "IGNORE_ASSERTS_UNTIL"
    )
    clamp_invalid_op = not args.allow_invalid_op

    def dep_list() -> list[str]:
        return [
            str(rtl_path / "aes_pkg.sv"),
            str(rtl_path / "aes_reg_pkg.sv"),
            str(rtl_path / "aes_sbox_canright_pkg.sv"),
            str(prim_path / "prim_util_pkg.sv"),
            pick_prim(prim_xilinx_path, "prim_xilinx_buf.sv", "prim_buf.sv"),
            pick_prim(prim_xilinx_path, "prim_xilinx_xor2.sv", "prim_xor2.sv"),
        ]

    if args.workdir:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        keep_workdir = True
    else:
        workdir = Path(tempfile.mkdtemp(prefix="opentitan-bmc-"))
        keep_workdir = args.keep_workdir

    try:
        cases_file = workdir / "pairwise-cases.tsv"
        with cases_file.open("w", encoding="utf-8") as handle:
            for impl in sorted(impl_list):
                impl_dir = workdir / impl
                impl_dir.mkdir(parents=True, exist_ok=True)

                src_ref = rtl_path / f"{impl_gold}.sv"
                src_impl = rtl_path / f"{impl}.sv"
                ref_module = impl_gold
                dut_module = impl
                extra_files: list[str] = [str(src_impl)]

                if "masked" in impl:
                    wrapper_out = impl_dir / "aes_sbox_masked_wrapper.sv"
                    replace_text(
                        wrapper_path,
                        wrapper_out,
                        [("aes_sbox_masked", impl)],
                    )
                    extra_files = [str(wrapper_out), str(src_impl)]
                    dut_module = f"{impl}_wrapper"

                if clamp_invalid_op:
                    ref_wrapper = f"{ref_module}_bmc_wrapper"
                    ref_wrapper_path = impl_dir / f"{ref_wrapper}.sv"
                    write_valid_op_wrapper(ref_wrapper_path, ref_wrapper, ref_module)
                    ref_module = ref_wrapper
                    extra_files.append(str(ref_wrapper_path))

                    dut_wrapper = f"{dut_module}_bmc_wrapper"
                    dut_wrapper_path = impl_dir / f"{dut_wrapper}.sv"
                    write_valid_op_wrapper(dut_wrapper_path, dut_wrapper, dut_module)
                    dut_module = dut_wrapper
                    extra_files.append(str(dut_wrapper_path))

                miter_module = f"aes_sbox_bmc_miter_{sanitize_identifier(impl)}"
                miter_path = impl_dir / f"{miter_module}.sv"
                write_bmc_miter(miter_path, miter_module, ref_module, dut_module)
                extra_files.append(str(miter_path))

                source_files = dep_list() + [str(src_ref)] + extra_files
                include_dirs = [str(prim_path), str(prim_xilinx_path), str(rtl_path)]

                policy = resolve_case_policy_for_impl(
                    impl, policy_by_impl, policy_pattern_rules, default_policy
                )
                timeout_cell = ""
                backend_cell = ""
                bound_cell = ""
                ignore_cell = ""
                assume_cell = ""
                allow_mc_cell = ""
                bmc_extra_args_cell = ""
                if policy is not None:
                    timeout_value = policy.get("timeout_secs")
                    bound_value = policy.get("bmc_bound")
                    ignore_value = policy.get("ignore_asserts_until")
                    backend_value = str(policy.get("backend_mode", "default"))
                    assume_value = str(policy.get("assume_known_inputs", "default"))
                    allow_value = str(policy.get("allow_multi_clock", "default"))
                    bmc_extra_args_value = str(policy.get("bmc_extra_args", ""))

                    timeout_cell = "" if timeout_value is None else str(timeout_value)
                    backend_cell = "" if backend_value == "default" else backend_value
                    bound_cell = "" if bound_value is None else str(bound_value)
                    ignore_cell = "" if ignore_value is None else str(ignore_value)
                    assume_cell = "" if assume_value == "default" else assume_value
                    allow_mc_cell = "" if allow_value == "default" else allow_value
                    bmc_extra_args_cell = bmc_extra_args_value

                handle.write(
                    "\t".join(
                        [
                            impl,
                            miter_module,
                            ";".join(source_files),
                            ";".join(include_dirs),
                            str(impl_dir),
                            timeout_cell,
                            backend_cell,
                            bound_cell,
                            ignore_cell,
                            assume_cell,
                            allow_mc_cell,
                            bmc_extra_args_cell,
                        ]
                    )
                    + "\n"
                )

        pairwise_runner = Path(__file__).resolve().with_name("run_pairwise_circt_bmc.py")
        if not pairwise_runner.is_file():
            print(f"missing pairwise runner: {pairwise_runner}", file=sys.stderr)
            return 1

        cmd = [
            sys.executable,
            str(pairwise_runner),
            "--cases-file",
            str(cases_file),
            "--suite-name",
            "opentitan",
            "--mode-label",
            mode_label,
            "--bound",
            str(bound),
            "--ignore-asserts-until",
            str(ignore_asserts_until),
            "--workdir",
            str(workdir),
            "--keep-workdir",
        ]
        if args.results_file:
            cmd += ["--results-file", args.results_file]
        if args.drop_remark_cases_file:
            cmd += ["--drop-remark-cases-file", args.drop_remark_cases_file]
        if args.drop_remark_reasons_file:
            cmd += ["--drop-remark-reasons-file", args.drop_remark_reasons_file]
        if args.timeout_reasons_file:
            cmd += ["--timeout-reasons-file", args.timeout_reasons_file]

        result = subprocess.run(cmd, check=False)
        return result.returncode
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

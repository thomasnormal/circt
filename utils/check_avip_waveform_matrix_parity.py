#!/usr/bin/env python3
"""Compare waveform artifacts for matching (avip,seed) rows across TSV matrices."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_int(text: str) -> int | None:
    value = text.strip()
    if not value:
        return None
    if value.startswith("-") and value != "-":
        if value[1:].isdigit():
            return int(value)
        return None
    if value.isdigit():
        return int(value)
    return None


def is_row_functional_pass(row: dict[str, str]) -> bool:
    if row.get("compile_status", "").strip() != "OK":
        return False
    if row.get("sim_status", "").strip() != "OK":
        return False

    sim_exit = parse_int(row.get("sim_exit", ""))
    if sim_exit is None or sim_exit != 0:
        return False

    uvm_fatal = parse_int(row.get("uvm_fatal", ""))
    uvm_error = parse_int(row.get("uvm_error", ""))
    if uvm_fatal is None or uvm_error is None:
        return False
    if uvm_fatal != 0 or uvm_error != 0:
        return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lhs-matrix", required=True, help="Path to first matrix TSV.")
    parser.add_argument("--rhs-matrix", required=True, help="Path to second matrix TSV.")
    parser.add_argument(
        "--lhs-label", default="lhs", help="Display label for first matrix lane."
    )
    parser.add_argument(
        "--rhs-label", default="rhs", help="Display label for second matrix lane."
    )
    parser.add_argument(
        "--lhs-vcd-column",
        default="vcd_file",
        help="Column name containing first lane VCD path (default: vcd_file).",
    )
    parser.add_argument(
        "--rhs-vcd-column",
        default="vcd_file",
        help="Column name containing second lane VCD path (default: vcd_file).",
    )
    parser.add_argument(
        "--compare-tool",
        default="",
        help=(
            "Waveform compare executable/script. "
            "Default: utils/compare_vcd_waveforms.py next to this script."
        ),
    )
    parser.add_argument(
        "--compare-arg",
        action="append",
        default=[],
        help=(
            "Pass-through argument chunk for compare tool (repeatable). "
            "Each entry is shell-split."
        ),
    )
    parser.add_argument(
        "--compare-nonfunctional",
        action="store_true",
        help="Also compare rows that fail functional checks.",
    )
    parser.add_argument(
        "--require-row-match",
        action="store_true",
        help="Treat missing (avip,seed) rows as failures.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero when any waveform mismatch is found.",
    )
    parser.add_argument(
        "--fail-on-missing-vcd",
        action="store_true",
        help="Exit non-zero when a comparable row is missing one or both VCD files.",
    )
    parser.add_argument(
        "--out-tsv",
        default="",
        help="Optional output TSV path for per-row waveform parity results.",
    )
    return parser.parse_args()


def read_matrix(path: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    if not path.is_file():
        fail(f"matrix file not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"matrix missing header row: {path}")
        rows: dict[tuple[str, str], dict[str, str]] = {}
        for row in reader:
            avip = (row.get("avip") or "").strip()
            seed = (row.get("seed") or "").strip()
            if not avip or not seed:
                continue
            key = (avip, seed)
            if key in rows:
                fail(f"duplicate matrix row key {avip}::{seed} in {path}")
            rows[key] = {k: (v or "").strip() for k, v in row.items()}
    return list(reader.fieldnames), rows


def build_compare_tool_path(arg: str) -> Path:
    if arg:
        path = Path(arg).resolve()
    else:
        path = (Path(__file__).resolve().parent / "compare_vcd_waveforms.py").resolve()
    if not path.is_file():
        fail(f"compare tool not found: {path}")
    return path


def is_missing_value(value: str) -> bool:
    text = value.strip()
    return not text or text in {"-", "?"}


def main() -> None:
    args = parse_args()

    lhs_matrix = Path(args.lhs_matrix).resolve()
    rhs_matrix = Path(args.rhs_matrix).resolve()
    compare_tool = build_compare_tool_path(args.compare_tool)

    lhs_cols, lhs_rows = read_matrix(lhs_matrix)
    rhs_cols, rhs_rows = read_matrix(rhs_matrix)
    if args.lhs_vcd_column not in lhs_cols:
        fail(f"lhs matrix missing column '{args.lhs_vcd_column}': {lhs_matrix}")
    if args.rhs_vcd_column not in rhs_cols:
        fail(f"rhs matrix missing column '{args.rhs_vcd_column}': {rhs_matrix}")

    compare_args: list[str] = []
    for chunk in args.compare_arg:
        compare_args.extend(shlex.split(chunk))

    keys = sorted(set(lhs_rows.keys()).union(rhs_rows.keys()))
    if not keys:
        fail("no rows found in matrices")

    out_rows: list[dict[str, str]] = []
    compared = 0
    mismatches = 0
    missing_vcd = 0
    missing_rows = 0
    skipped_nonfunctional = 0

    for avip, seed in keys:
        lhs_row = lhs_rows.get((avip, seed))
        rhs_row = rhs_rows.get((avip, seed))

        status = "unknown"
        note = ""
        compare_rc = "-"
        lhs_vcd = "-"
        rhs_vcd = "-"
        lhs_functional = "0"
        rhs_functional = "0"

        if lhs_row is None or rhs_row is None:
            missing_rows += 1
            status = "row_missing"
            note = (
                f"missing_in_{args.lhs_label}"
                if lhs_row is None
                else f"missing_in_{args.rhs_label}"
            )
            if not args.require_row_match:
                status = "row_missing_ignored"
        else:
            lhs_functional = "1" if is_row_functional_pass(lhs_row) else "0"
            rhs_functional = "1" if is_row_functional_pass(rhs_row) else "0"
            if (
                not args.compare_nonfunctional
                and (lhs_functional != "1" or rhs_functional != "1")
            ):
                skipped_nonfunctional += 1
                status = "skipped_nonfunctional"
            else:
                lhs_vcd = lhs_row.get(args.lhs_vcd_column, "").strip()
                rhs_vcd = rhs_row.get(args.rhs_vcd_column, "").strip()
                if is_missing_value(lhs_vcd) or is_missing_value(rhs_vcd):
                    missing_vcd += 1
                    status = "missing_vcd"
                    note = (
                        f"{args.lhs_label}_missing"
                        if is_missing_value(lhs_vcd)
                        else f"{args.rhs_label}_missing"
                    )
                else:
                    lhs_vcd_path = Path(lhs_vcd).resolve()
                    rhs_vcd_path = Path(rhs_vcd).resolve()
                    if not lhs_vcd_path.is_file() or not rhs_vcd_path.is_file():
                        missing_vcd += 1
                        status = "missing_vcd"
                        if not lhs_vcd_path.is_file():
                            note = f"{args.lhs_label}_path_missing"
                        elif not rhs_vcd_path.is_file():
                            note = f"{args.rhs_label}_path_missing"
                    else:
                        compared += 1
                        cmd = [
                            sys.executable,
                            str(compare_tool),
                            str(lhs_vcd_path),
                            str(rhs_vcd_path),
                            "--label-lhs",
                            args.lhs_label,
                            "--label-rhs",
                            args.rhs_label,
                            "--fail-on-mismatch",
                            *compare_args,
                        ]
                        proc = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=False,
                        )
                        compare_rc = str(proc.returncode)
                        if proc.returncode == 0:
                            status = "match"
                            note = proc.stdout.strip().splitlines()[-1] if proc.stdout else ""
                        else:
                            mismatches += 1
                            status = "mismatch"
                            detail = proc.stderr.strip() or proc.stdout.strip()
                            note = detail.splitlines()[0] if detail else "compare_failed"

        out_rows.append(
            {
                "avip": avip,
                "seed": seed,
                "status": status,
                "lhs_functional": lhs_functional,
                "rhs_functional": rhs_functional,
                "lhs_vcd": lhs_vcd,
                "rhs_vcd": rhs_vcd,
                "compare_rc": compare_rc,
                "note": note,
            }
        )

    if args.out_tsv:
        out_path = Path(args.out_tsv).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    "avip",
                    "seed",
                    "status",
                    "lhs_functional",
                    "rhs_functional",
                    "lhs_vcd",
                    "rhs_vcd",
                    "compare_rc",
                    "note",
                ]
            )
            for row in out_rows:
                writer.writerow(
                    [
                        row["avip"],
                        row["seed"],
                        row["status"],
                        row["lhs_functional"],
                        row["rhs_functional"],
                        row["lhs_vcd"],
                        row["rhs_vcd"],
                        row["compare_rc"],
                        row["note"],
                    ]
                )

    print(f"rows_total={len(out_rows)}")
    print(f"rows_compared={compared}")
    print(f"rows_mismatch={mismatches}")
    print(f"rows_missing_vcd={missing_vcd}")
    print(f"rows_missing_matrix={missing_rows}")
    print(f"rows_skipped_nonfunctional={skipped_nonfunctional}")

    if args.fail_on_mismatch and mismatches > 0:
        fail(f"waveform parity mismatches detected: {mismatches}")
    if args.fail_on_missing_vcd and missing_vcd > 0:
        fail(f"missing VCD files for comparable rows: {missing_vcd}")
    if args.require_row_match and missing_rows > 0:
        fail(f"matrix row mismatches detected: {missing_rows}")


if __name__ == "__main__":
    main()

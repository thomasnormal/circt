#!/usr/bin/env python3
"""Compare CIRCT and Xcelium AVIP matrix TSVs and apply parity gates."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


RowKey = Tuple[str, str]
RowMap = Dict[RowKey, dict[str, str]]


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("circt_matrix", help="Path to CIRCT matrix.tsv.")
    parser.add_argument("xcelium_matrix", help="Path to Xcelium matrix.tsv.")
    parser.add_argument(
        "--out-tsv",
        default="",
        help="Optional output TSV path for per-AVIP parity summary.",
    )
    parser.add_argument(
        "--fail-on-functional",
        action="store_true",
        help="Exit non-zero if any functional mismatch/failure row is found.",
    )
    parser.add_argument(
        "--fail-on-coverage",
        action="store_true",
        help="Exit non-zero if any coverage-below-baseline row is found.",
    )
    parser.add_argument(
        "--require-row-match",
        action="store_true",
        help="Treat missing rows in either matrix as functional failures.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"matrix file not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"matrix missing header: {path}")
        rows = list(reader)
    if not rows:
        fail(f"matrix has no data rows: {path}")
    required = {"avip", "seed"}
    missing = required.difference(rows[0].keys())
    if missing:
        fail(f"matrix missing required columns {sorted(missing)}: {path}")
    return rows


def build_row_map(rows: Iterable[dict[str, str]], source: str) -> RowMap:
    out: RowMap = {}
    for row in rows:
        avip = row.get("avip", "").strip()
        seed = row.get("seed", "").strip()
        if not avip or not seed:
            fail(f"{source}: encountered row with empty avip/seed")
        key = (avip, seed)
        if key in out:
            fail(f"{source}: duplicate row for avip={avip} seed={seed}")
        out[key] = row
    return out


def parse_int(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    if text.startswith("-") and text != "-":
        if text[1:].isdigit():
            return int(text)
        return None
    if text.isdigit():
        return int(text)
    return None


def parse_float(value: str) -> float | None:
    text = value.strip()
    if not text or text in {"-", "?"}:
        return None
    try:
        return float(text)
    except ValueError:
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


def coverage_pass(circt_row: dict[str, str], xcelium_row: dict[str, str]) -> tuple[bool, float | None, float | None]:
    circt_cov1 = parse_float(circt_row.get("cov_1_pct", ""))
    circt_cov2 = parse_float(circt_row.get("cov_2_pct", ""))
    xcelium_cov1 = parse_float(xcelium_row.get("cov_1_pct", ""))
    xcelium_cov2 = parse_float(xcelium_row.get("cov_2_pct", ""))

    cov1_delta: float | None = None
    cov2_delta: float | None = None

    if xcelium_cov1 is not None:
        if circt_cov1 is None:
            return False, cov1_delta, cov2_delta
        cov1_delta = circt_cov1 - xcelium_cov1
        if cov1_delta < 0:
            return False, cov1_delta, cov2_delta

    if xcelium_cov2 is not None:
        if circt_cov2 is None:
            return False, cov1_delta, cov2_delta
        cov2_delta = circt_cov2 - xcelium_cov2
        if cov2_delta < 0:
            return False, cov1_delta, cov2_delta

    return True, cov1_delta, cov2_delta


def write_summary_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "avip",
                "rows_total",
                "rows_missing_circt",
                "rows_missing_xcelium",
                "functional_fail_rows",
                "coverage_fail_rows",
                "min_cov1_delta_pct",
                "min_cov2_delta_pct",
                "functional_pass",
                "coverage_pass",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["avip"],
                    row["rows_total"],
                    row["rows_missing_circt"],
                    row["rows_missing_xcelium"],
                    row["functional_fail_rows"],
                    row["coverage_fail_rows"],
                    row["min_cov1_delta_pct"],
                    row["min_cov2_delta_pct"],
                    row["functional_pass"],
                    row["coverage_pass"],
                ]
            )


def format_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()

    circt_path = Path(args.circt_matrix).resolve()
    xcelium_path = Path(args.xcelium_matrix).resolve()

    circt_rows = read_tsv(circt_path)
    xcelium_rows = read_tsv(xcelium_path)
    circt_by_key = build_row_map(circt_rows, "circt")
    xcelium_by_key = build_row_map(xcelium_rows, "xcelium")

    keys = sorted(set(circt_by_key.keys()).union(xcelium_by_key.keys()))
    if not keys:
        fail("no comparable avip/seed rows found")

    functional_fail_rows = 0
    coverage_fail_rows = 0
    missing_circt_rows = 0
    missing_xcelium_rows = 0

    per_avip = defaultdict(
        lambda: {
            "rows_total": 0,
            "rows_missing_circt": 0,
            "rows_missing_xcelium": 0,
            "functional_fail_rows": 0,
            "coverage_fail_rows": 0,
            "min_cov1_delta": None,
            "min_cov2_delta": None,
        }
    )

    for key in keys:
        avip, _seed = key
        stats = per_avip[avip]
        stats["rows_total"] += 1

        circt_row = circt_by_key.get(key)
        xcelium_row = xcelium_by_key.get(key)
        row_functional_fail = False
        row_coverage_fail = False

        if circt_row is None:
            missing_circt_rows += 1
            stats["rows_missing_circt"] += 1
            row_functional_fail = args.require_row_match
            row_coverage_fail = args.require_row_match
        if xcelium_row is None:
            missing_xcelium_rows += 1
            stats["rows_missing_xcelium"] += 1
            row_functional_fail = row_functional_fail or args.require_row_match
            row_coverage_fail = row_coverage_fail or args.require_row_match

        if circt_row is not None and xcelium_row is not None:
            if not is_row_functional_pass(circt_row):
                row_functional_fail = True
                row_coverage_fail = True
            else:
                passed, cov1_delta, cov2_delta = coverage_pass(circt_row, xcelium_row)
                if not passed:
                    row_coverage_fail = True

                if cov1_delta is not None:
                    current = stats["min_cov1_delta"]
                    stats["min_cov1_delta"] = cov1_delta if current is None else min(current, cov1_delta)
                if cov2_delta is not None:
                    current = stats["min_cov2_delta"]
                    stats["min_cov2_delta"] = cov2_delta if current is None else min(current, cov2_delta)

        if row_functional_fail:
            functional_fail_rows += 1
            stats["functional_fail_rows"] += 1
        if row_coverage_fail:
            coverage_fail_rows += 1
            stats["coverage_fail_rows"] += 1

    summary_rows: list[dict[str, str]] = []
    for avip in sorted(per_avip.keys()):
        stats = per_avip[avip]
        functional_pass = stats["functional_fail_rows"] == 0
        coverage_pass_all = stats["coverage_fail_rows"] == 0
        summary_rows.append(
            {
                "avip": avip,
                "rows_total": str(stats["rows_total"]),
                "rows_missing_circt": str(stats["rows_missing_circt"]),
                "rows_missing_xcelium": str(stats["rows_missing_xcelium"]),
                "functional_fail_rows": str(stats["functional_fail_rows"]),
                "coverage_fail_rows": str(stats["coverage_fail_rows"]),
                "min_cov1_delta_pct": format_delta(stats["min_cov1_delta"]),
                "min_cov2_delta_pct": format_delta(stats["min_cov2_delta"]),
                "functional_pass": "1" if functional_pass else "0",
                "coverage_pass": "1" if coverage_pass_all else "0",
            }
        )

    if args.out_tsv:
        write_summary_tsv(Path(args.out_tsv).resolve(), summary_rows)

    print(f"rows_total={len(keys)}")
    print(f"rows_missing_circt={missing_circt_rows}")
    print(f"rows_missing_xcelium={missing_xcelium_rows}")
    print(f"functional_fail_rows={functional_fail_rows}")
    print(f"coverage_fail_rows={coverage_fail_rows}")
    print(f"unique_avips={len(summary_rows)}")

    for row in summary_rows:
        print(
            "avip_summary "
            f"avip={row['avip']} rows={row['rows_total']} "
            f"functional_fail_rows={row['functional_fail_rows']} "
            f"coverage_fail_rows={row['coverage_fail_rows']} "
            f"min_cov1_delta_pct={row['min_cov1_delta_pct']} "
            f"min_cov2_delta_pct={row['min_cov2_delta_pct']}"
        )

    if args.fail_on_functional and functional_fail_rows > 0:
        fail(f"functional parity gate failed: functional_fail_rows={functional_fail_rows}")
    if args.fail_on_coverage and coverage_fail_rows > 0:
        fail(f"coverage parity gate failed: coverage_fail_rows={coverage_fail_rows}")


if __name__ == "__main__":
    main()

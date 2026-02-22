#!/usr/bin/env python3
"""Check interpret-vs-compile parity for circt-sim AVIP matrix artifacts."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


KEY_FIELDS = (
    "compile_status",
    "sim_status",
    "sim_exit",
    "sim_time_fs",
    "uvm_fatal",
    "uvm_error",
    "cov_1_pct",
    "cov_2_pct",
)


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interpret-matrix",
        required=True,
        help="TSV matrix artifact from CIRCT_SIM_MODE=interpret run.",
    )
    parser.add_argument(
        "--compile-matrix",
        required=True,
        help="TSV matrix artifact from CIRCT_SIM_MODE=compile run.",
    )
    parser.add_argument(
        "--allowlist-file",
        default="",
        help=(
            "Optional allowlist file. Each non-comment line is exact:<token>, "
            "prefix:<prefix>, regex:<pattern>, or bare exact. Tokens may be "
            "<avip>, <avip>::<seed>, <avip>::<kind>, or <avip>::<seed>::<kind>."
        ),
    )
    parser.add_argument(
        "--out-parity-tsv",
        default="",
        help="Optional output TSV path for parity drift rows.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero when non-allowlisted parity mismatches are detected.",
    )
    return parser.parse_args()


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


def read_matrix(path: Path, lane_name: str) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        fail(f"{lane_name} matrix file not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"{lane_name} matrix missing header row: {path}")
        required = {"avip", "seed", *KEY_FIELDS}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            fail(
                f"{lane_name} matrix missing required columns {missing}: {path} "
                f"(found: {reader.fieldnames})"
            )
        out: dict[tuple[str, str], dict[str, str]] = {}
        for idx, row in enumerate(reader, start=2):
            avip = (row.get("avip") or "").strip()
            seed = (row.get("seed") or "").strip()
            if not avip or not seed:
                continue
            key = (avip, seed)
            if key in out:
                fail(f"duplicate matrix row key '{avip}::{seed}' in {path} row {idx}")
            out[key] = {field: (row.get(field) or "").strip() for field in KEY_FIELDS}
    return out


def emit_parity_tsv(
    path: Path, rows: list[tuple[str, str, str, str, str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["avip", "seed", "kind", "interpret", "compile", "allowlisted"])
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    interpret_path = Path(args.interpret_matrix).resolve()
    compile_path = Path(args.compile_matrix).resolve()

    allow_exact: set[str] = set()
    allow_prefix: list[str] = []
    allow_regex: list[re.Pattern[str]] = []
    if args.allowlist_file:
        allow_exact, allow_prefix, allow_regex = load_allowlist(
            Path(args.allowlist_file).resolve()
        )

    interpret_rows = read_matrix(interpret_path, "interpret")
    compile_rows = read_matrix(compile_path, "compile")

    parity_rows: list[tuple[str, str, str, str, str, str]] = []
    non_allowlisted_rows: list[tuple[str, str, str, str, str, str]] = []

    def add_row(
        avip: str, seed: str, kind: str, interpret_value: str, compile_value: str
    ) -> None:
        tokens = (
            f"{avip}::{seed}::{kind}",
            f"{avip}::{kind}",
            f"{avip}::{seed}",
            avip,
        )
        allowlisted = any(
            is_allowlisted(token, allow_exact, allow_prefix, allow_regex)
            for token in tokens
        )
        row = (
            avip,
            seed,
            kind,
            interpret_value,
            compile_value,
            "1" if allowlisted else "0",
        )
        parity_rows.append(row)
        if not allowlisted:
            non_allowlisted_rows.append(row)

    interpret_keys = set(interpret_rows.keys())
    compile_keys = set(compile_rows.keys())
    for avip, seed in sorted(interpret_keys - compile_keys):
        add_row(avip, seed, "missing_in_compile", "present", "absent")
    for avip, seed in sorted(compile_keys - interpret_keys):
        add_row(avip, seed, "missing_in_interpret", "absent", "present")

    for avip, seed in sorted(interpret_keys.intersection(compile_keys)):
        interpret_row = interpret_rows[(avip, seed)]
        compile_row = compile_rows[(avip, seed)]
        for field in KEY_FIELDS:
            interpret_value = interpret_row.get(field, "")
            compile_value = compile_row.get(field, "")
            if interpret_value != compile_value:
                add_row(avip, seed, field, interpret_value, compile_value)

    if args.out_parity_tsv:
        emit_parity_tsv(Path(args.out_parity_tsv).resolve(), parity_rows)

    if non_allowlisted_rows:
        sample = ", ".join(
            f"{avip}/{seed}:{kind}"
            for avip, seed, kind, _, _, _ in non_allowlisted_rows[:6]
        )
        if len(non_allowlisted_rows) > 6:
            sample += ", ..."
        message = (
            "circt-sim mode parity mismatches detected: "
            f"rows={len(non_allowlisted_rows)} sample=[{sample}] "
            f"interpret={interpret_path} compile={compile_path}"
        )
        if args.fail_on_mismatch:
            print(message, file=sys.stderr)
            raise SystemExit(1)
        print(f"warning: {message}", file=sys.stderr)
        return

    print(
        "circt-sim mode parity check passed: "
        f"rows_interpret={len(interpret_rows)} rows_compile={len(compile_rows)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

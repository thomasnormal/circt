#!/usr/bin/env python3
"""Plan native mutation labels for MCY example flows."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

NATIVE_OPS_ALL = [
    "EQ_TO_NEQ",
    "NEQ_TO_EQ",
    "LT_TO_LE",
    "GT_TO_GE",
    "LE_TO_LT",
    "GE_TO_GT",
    "AND_TO_OR",
    "OR_TO_AND",
    "XOR_TO_OR",
    "UNARY_NOT_DROP",
    "CONST0_TO_1",
    "CONST1_TO_0",
]

OP_PATTERNS = {
    "EQ_TO_NEQ": r"==",
    "NEQ_TO_EQ": r"!=",
    "LT_TO_LE": r"(?<![<>=!])<(?![<>=])",
    "GT_TO_GE": r"(?<![<>=!])>(?![<>=])",
    "LE_TO_LT": r"<=",
    "GE_TO_GT": r">=",
    "AND_TO_OR": r"&&",
    "OR_TO_AND": r"\|\|",
    "XOR_TO_OR": r"\^",
    "UNARY_NOT_DROP": r"!\s*(?=[A-Za-z_(])",
    "CONST0_TO_1": r"1'b0|1'd0",
    "CONST1_TO_0": r"1'b1|1'd1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ops-csv", default="")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def parse_ops_csv(ops_csv: str) -> list[str]:
    if not ops_csv.strip():
        return list(NATIVE_OPS_ALL)
    requested = [tok.strip() for tok in ops_csv.split(",") if tok.strip()]
    unknown = [tok for tok in requested if tok not in NATIVE_OPS_ALL]
    if unknown:
        raise ValueError("unsupported native ops: " + ", ".join(sorted(set(unknown))))
    return requested


def compute_applicable_ops(design_text: str) -> list[str]:
    applicable: list[str] = []
    for op in NATIVE_OPS_ALL:
        pattern = OP_PATTERNS[op]
        if re.search(pattern, design_text):
            applicable.append(op)
    return applicable


def order_ops(base_ops: list[str], applicable: list[str]) -> list[str]:
    if not applicable:
        return list(base_ops)
    ordered: list[str] = []
    for op in applicable:
        if op in base_ops and op not in ordered:
            ordered.append(op)
    for op in base_ops:
        if op not in ordered:
            ordered.append(op)
    return ordered


def emit_plan(ops: list[str], count: int, seed: int) -> list[str]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if not ops:
        raise ValueError("native mutation operator set must not be empty")
    if count == 0:
        return []
    seed_offset = seed % len(ops)
    lines: list[str] = []
    for mid in range(1, count + 1):
        op_idx = (seed_offset + mid - 1) % len(ops)
        lines.append(f"{mid} NATIVE_{ops[op_idx]}")
    return lines


def main() -> int:
    args = parse_args()
    design_path = Path(args.design)
    out_path = Path(args.out)

    design_text = design_path.read_text(encoding="utf-8")
    base_ops = parse_ops_csv(args.ops_csv)
    applicable = compute_applicable_ops(design_text)
    ordered_ops = order_ops(base_ops, applicable)
    lines = emit_plan(ordered_ops, args.count, args.seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    out_path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

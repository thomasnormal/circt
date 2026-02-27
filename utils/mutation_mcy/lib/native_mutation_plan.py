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


def build_code_mask(text: str) -> list[bool]:
    mask = [True] * len(text)
    state = "normal"
    escape = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if state == "normal":
            if ch == "/" and i + 1 < n and text[i + 1] == "/":
                mask[i] = False
                mask[i + 1] = False
                i += 2
                state = "line_comment"
                continue
            if ch == "/" and i + 1 < n and text[i + 1] == "*":
                mask[i] = False
                mask[i + 1] = False
                i += 2
                state = "block_comment"
                continue
            if ch == '"':
                mask[i] = False
                state = "string"
                escape = False
                i += 1
                continue
            i += 1
            continue
        if state == "line_comment":
            if ch != "\n":
                mask[i] = False
            else:
                state = "normal"
            i += 1
            continue
        if state == "block_comment":
            mask[i] = False
            if ch == "*" and i + 1 < n and text[i + 1] == "/":
                mask[i + 1] = False
                i += 2
                state = "normal"
                continue
            i += 1
            continue
        if state == "string":
            mask[i] = False
            if escape:
                escape = False
                i += 1
                continue
            if ch == "\\":
                escape = True
                i += 1
                continue
            if ch == '"':
                state = "normal"
            i += 1
            continue
    return mask


def is_code_span(mask: list[bool], start: int, end: int) -> bool:
    if start < 0 or end > len(mask) or start >= end:
        return False
    return all(mask[i] for i in range(start, end))


def count_literal_token(text: str, token: str, mask: list[bool]) -> int:
    if not token:
        return 0
    pos = 0
    count = 0
    while True:
        pos = text.find(token, pos)
        if pos < 0:
            return count
        if is_code_span(mask, pos, pos + len(token)):
            count += 1
        pos += len(token)


def count_relational_comparator_token(text: str, token: str, mask: list[bool]) -> int:
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    saw_plain_assign = False
    count = 0
    i = 0
    n = len(text)
    while i + 1 < n:
        if not mask[i]:
            i += 1
            continue
        ch = text[i]
        nxt = text[i + 1] if mask[i + 1] else ""

        if ch == ';':
            saw_plain_assign = False
            i += 1
            continue
        if ch == '(':
            paren_depth += 1
        elif ch == ')':
            paren_depth = max(paren_depth - 1, 0)
        elif ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth = max(bracket_depth - 1, 0)
        elif ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth = max(brace_depth - 1, 0)

        if ch == '=':
            prev = text[i - 1] if i > 0 and mask[i - 1] else ""
            if prev not in ("=", "!", "<", ">") and nxt != "=":
                saw_plain_assign = True

        if is_code_span(mask, i, i + len(token)) and text.startswith(token, i):
            prev = text[i - 1] if i > 0 and mask[i - 1] else ""
            if token == "<=" and prev == "<":
                i += 1
                continue
            if token == ">=" and prev == ">":
                i += 1
                continue
            if paren_depth > 0 or bracket_depth > 0 or brace_depth > 0 or saw_plain_assign:
                count += 1
        i += 1
    return count


def count_native_mutation_sites(design_text: str, op: str, mask: list[bool]) -> int:
    if op == "LE_TO_LT":
        return count_relational_comparator_token(design_text, "<=", mask)
    if op == "GE_TO_GT":
        return count_relational_comparator_token(design_text, ">=", mask)
    pattern = OP_PATTERNS[op]
    if op in ("EQ_TO_NEQ", "NEQ_TO_EQ", "AND_TO_OR", "OR_TO_AND", "XOR_TO_OR"):
        return count_literal_token(design_text, pattern.replace("\\", ""), mask)
    return sum(1 for m in re.finditer(pattern, design_text) if is_code_span(mask, m.start(), m.end()))


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
    mask = build_code_mask(design_text)
    applicable: list[str] = []
    for op in NATIVE_OPS_ALL:
        if count_native_mutation_sites(design_text, op, mask) > 0:
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


def emit_plan(
    ops: list[str], site_counts: dict[str, int], count: int, seed: int
) -> list[str]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if not ops:
        raise ValueError("native mutation operator set must not be empty")
    if count == 0:
        return []
    seed_offset = seed % len(ops)
    lines: list[str] = []
    for mid in range(1, count + 1):
        rank = seed_offset + mid - 1
        op_idx = rank % len(ops)
        cycle = rank // len(ops)
        op = ops[op_idx]
        site_count = max(1, int(site_counts.get(op, 0)))
        site_index = ((seed + cycle) % site_count) + 1
        label = f"NATIVE_{op}"
        if site_count > 1 or cycle > 0:
            label += f"@{site_index}"
        lines.append(f"{mid} {label}")
    return lines


def main() -> int:
    args = parse_args()
    design_path = Path(args.design)
    out_path = Path(args.out)

    design_text = design_path.read_text(encoding="utf-8")
    base_ops = parse_ops_csv(args.ops_csv)
    mask = build_code_mask(design_text)
    applicable = compute_applicable_ops(design_text)
    ordered_ops = order_ops(base_ops, applicable)
    site_counts = {op: count_native_mutation_sites(design_text, op, mask) for op in ordered_ops}
    lines = emit_plan(ordered_ops, site_counts, args.count, args.seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    out_path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    "IF_COND_NEGATE",
    "IF_ELSE_SWAP_ARMS",
    "UNARY_NOT_DROP",
    "CONST0_TO_1",
    "CONST1_TO_0",
    "POSEDGE_TO_NEGEDGE",
    "NEGEDGE_TO_POSEDGE",
    "MUX_SWAP_ARMS",
    "INC_TO_DEC",
    "DEC_TO_INC",
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
    "POSEDGE_TO_NEGEDGE": r"\bposedge\b",
    "NEGEDGE_TO_POSEDGE": r"\bnegedge\b",
    "INC_TO_DEC": r"\+\+",
    "DEC_TO_INC": r"--",
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


def is_code_at(mask: list[bool], pos: int) -> bool:
    return 0 <= pos < len(mask) and mask[pos]


def is_identifier_body(ch: str) -> bool:
    return ch.isalnum() or ch in ("_", "$")


def find_prev_code_nonspace(text: str, mask: list[bool], pos: int) -> int:
    i = pos
    while i > 0:
        i -= 1
        if not is_code_at(mask, i):
            continue
        if text[i].isspace():
            continue
        return i
    return -1


def find_next_code_nonspace(text: str, mask: list[bool], pos: int) -> int:
    i = pos
    n = len(text)
    while i < n:
        if not is_code_at(mask, i):
            i += 1
            continue
        if text[i].isspace():
            i += 1
            continue
        return i
    return -1


def is_operand_end_char(ch: str) -> bool:
    return ch.isalnum() or ch in ("_", ")", "]", "}", "'")


def is_operand_start_char(ch: str) -> bool:
    return ch.isalnum() or ch in ("_", "(", "[", "{", "'", "~", "!", "$")


def find_statement_start(text: str, mask: list[bool], pos: int) -> int:
    i = pos
    while i > 0:
        i -= 1
        if not is_code_at(mask, i):
            continue
        if text[i] in (";", "\n", "{", "}"):
            return i + 1
    return 0


def statement_has_assignment_disqualifier(
    text: str, mask: list[bool], stmt_start: int, pos: int, include_assign: bool = True
) -> bool:
    disqualifiers = {
        "parameter",
        "localparam",
        "typedef",
        "input",
        "output",
        "inout",
        "wire",
        "logic",
        "reg",
        "bit",
        "byte",
        "shortint",
        "int",
        "longint",
        "integer",
        "time",
        "realtime",
        "real",
        "string",
        "enum",
        "struct",
        "union",
        "genvar",
        "module",
        "interface",
        "package",
        "class",
        "function",
        "task",
    }
    if include_assign:
        disqualifiers.add("assign")
    i = max(0, stmt_start)
    end = min(pos, len(text))
    while i < end:
        if not is_code_at(mask, i) or not (text[i].isalpha() or text[i] == "_"):
            i += 1
            continue
        start = i
        i += 1
        while i < end and is_code_at(mask, i) and (text[i].isalnum() or text[i] in ("_", "$")):
            i += 1
        token = text[start:i].lower()
        if token in disqualifiers:
            return True
    return False


def statement_has_assignment_before(text: str, mask: list[bool], stmt_start: int, pos: int) -> bool:
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    i = max(0, stmt_start)
    end = min(pos, len(text))
    while i < end:
        if not is_code_at(mask, i):
            i += 1
            continue
        ch = text[i]
        if ch == "(":
            paren_depth += 1
            i += 1
            continue
        if ch == ")":
            paren_depth = max(paren_depth - 1, 0)
            i += 1
            continue
        if ch == "[":
            bracket_depth += 1
            i += 1
            continue
        if ch == "]":
            bracket_depth = max(bracket_depth - 1, 0)
            i += 1
            continue
        if ch == "{":
            brace_depth += 1
            i += 1
            continue
        if ch == "}":
            brace_depth = max(brace_depth - 1, 0)
            i += 1
            continue
        if paren_depth > 0 or bracket_depth > 0 or brace_depth > 0:
            i += 1
            continue

        if ch == "=":
            prev = text[i - 1] if i > 0 and is_code_at(mask, i - 1) else ""
            nxt = text[i + 1] if i + 1 < len(text) and is_code_at(mask, i + 1) else ""
            if prev in ("=", "!", "<", ">") or nxt in ("=", ">"):
                i += 1
                continue
            return True

        if is_code_span(mask, i, i + 2) and text.startswith("<=", i):
            prev = text[i - 1] if i > 0 and is_code_at(mask, i - 1) else ""
            nxt = text[i + 2] if i + 2 < len(text) and is_code_at(mask, i + 2) else ""
            if prev in ("<", "=", "!", ">") or nxt in ("=", ">"):
                i += 1
                continue
            return True
        i += 1
    return False


def _skip_code_ws(text: str, mask: list[bool], i: int, end: int) -> int:
    while i < end:
        if not is_code_at(mask, i):
            i += 1
            continue
        if text[i].isspace():
            i += 1
            continue
        break
    return i


def _parse_identifier_token(text: str, mask: list[bool], i: int, end: int) -> tuple[int, int]:
    if i >= end or not is_code_at(mask, i):
        return (-1, -1)
    ch = text[i]
    if not (ch.isalpha() or ch == "_"):
        return (-1, -1)
    start = i
    i += 1
    while i < end and is_code_at(mask, i) and (text[i].isalnum() or text[i] in ("_", "$")):
        i += 1
    return (start, i)


def _skip_balanced(text: str, mask: list[bool], i: int, end: int, open_ch: str, close_ch: str) -> int:
    if i >= end or not is_code_at(mask, i) or text[i] != open_ch:
        return i
    depth = 0
    while i < end:
        if not is_code_at(mask, i):
            i += 1
            continue
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return end


def statement_looks_like_typed_declaration(text: str, mask: list[bool], stmt_start: int, pos: int) -> bool:
    end = min(pos, len(text))
    i = _skip_code_ws(text, mask, max(0, stmt_start), end)
    first_start, first_end = _parse_identifier_token(text, mask, i, end)
    if first_start < 0:
        return False
    first_token = text[first_start:first_end].lower()
    if first_token in {"assign", "if", "for", "while", "case", "foreach", "return", "begin", "end"}:
        return False
    i = first_end

    while True:
        i = _skip_code_ws(text, mask, i, end)
        if i + 1 < end and is_code_span(mask, i, i + 2) and text.startswith("::", i):
            i = _skip_code_ws(text, mask, i + 2, end)
            _, i = _parse_identifier_token(text, mask, i, end)
            if i < 0:
                return False
            continue
        if i < end and is_code_at(mask, i) and text[i] == "#":
            i = _skip_code_ws(text, mask, i + 1, end)
            i = _skip_balanced(text, mask, i, end, "(", ")")
            continue
        if i < end and is_code_at(mask, i) and text[i] == "[":
            i = _skip_balanced(text, mask, i, end, "[", "]")
            continue
        break

    i = _skip_code_ws(text, mask, i, end)
    second_start, _ = _parse_identifier_token(text, mask, i, end)
    if second_start < 0:
        return False
    prev_sig = find_prev_code_nonspace(text, mask, second_start)
    if prev_sig >= 0 and text[prev_sig] == ".":
        return False
    return True


def find_matching_ternary_colon(text: str, mask: list[bool], question_pos: int) -> int:
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    nested_ternary = 0
    i = question_pos + 1
    n = len(text)
    while i < n:
        if not is_code_at(mask, i):
            i += 1
            continue
        ch = text[i]
        if ch == "(":
            paren_depth += 1
            i += 1
            continue
        if ch == ")":
            if paren_depth == 0:
                return -1
            paren_depth -= 1
            i += 1
            continue
        if ch == "[":
            bracket_depth += 1
            i += 1
            continue
        if ch == "]":
            if bracket_depth == 0:
                return -1
            bracket_depth -= 1
            i += 1
            continue
        if ch == "{":
            brace_depth += 1
            i += 1
            continue
        if ch == "}":
            if brace_depth == 0:
                return -1
            brace_depth -= 1
            i += 1
            continue
        if paren_depth > 0 or bracket_depth > 0 or brace_depth > 0:
            i += 1
            continue
        if ch == "?":
            nested_ternary += 1
            i += 1
            continue
        if ch == ":":
            if nested_ternary == 0:
                return i
            nested_ternary -= 1
            i += 1
            continue
        if ch in (";", ","):
            return -1
        i += 1
    return -1


def find_matching_paren(text: str, mask: list[bool], open_pos: int) -> int:
    if open_pos < 0 or open_pos >= len(text):
        return -1
    if not is_code_at(mask, open_pos) or text[open_pos] != "(":
        return -1
    depth = 0
    i = open_pos
    n = len(text)
    while i < n:
        if not is_code_at(mask, i):
            i += 1
            continue
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
            if depth < 0:
                return -1
        i += 1
    return -1


def count_if_cond_negate_sites(text: str, mask: list[bool]) -> int:
    count = 0
    i = 0
    n = len(text)
    while i + 1 < n:
        if not is_code_span(mask, i, i + 2) or not text.startswith("if", i):
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(mask, i - 1) else ""
        nxt = text[i + 2] if i + 2 < n and is_code_at(mask, i + 2) else ""
        if (prev and is_identifier_body(prev)) or (nxt and is_identifier_body(nxt)):
            i += 1
            continue
        j = i + 2
        while j < n and ((not is_code_at(mask, j)) or text[j].isspace()):
            j += 1
        if j >= n or not is_code_at(mask, j) or text[j] != "(":
            i += 1
            continue
        k = find_matching_paren(text, mask, j)
        if k < 0:
            i += 1
            continue
        count += 1
        i += 1
    return count


def match_keyword_token(text: str, mask: list[bool], pos: int, keyword: str) -> bool:
    if pos < 0 or not keyword:
        return False
    klen = len(keyword)
    if pos + klen > len(text):
        return False
    if not is_code_span(mask, pos, pos + klen) or not text.startswith(keyword, pos):
        return False
    prev = text[pos - 1] if pos > 0 and is_code_at(mask, pos - 1) else ""
    nxt = text[pos + klen] if pos + klen < len(text) and is_code_at(mask, pos + klen) else ""
    prev_boundary = (not prev) or (not is_identifier_body(prev))
    next_boundary = (not nxt) or (not is_identifier_body(nxt))
    return prev_boundary and next_boundary


def find_matching_begin_end(text: str, mask: list[bool], begin_pos: int) -> int:
    if not match_keyword_token(text, mask, begin_pos, "begin"):
        return -1
    depth = 0
    i = begin_pos
    n = len(text)
    while i < n:
        if not is_code_at(mask, i):
            i += 1
            continue
        if match_keyword_token(text, mask, i, "begin"):
            depth += 1
            i += len("begin")
            continue
        if match_keyword_token(text, mask, i, "end"):
            depth -= 1
            i += len("end")
            if depth == 0:
                return i
            if depth < 0:
                return -1
            continue
        i += 1
    return -1


def find_if_else_branch_end(text: str, mask: list[bool], branch_start: int) -> int:
    i = find_next_code_nonspace(text, mask, branch_start)
    if i < 0:
        return -1

    # Keep deterministic semantics by skipping ambiguous dangling-else forms.
    if match_keyword_token(text, mask, i, "if"):
        return -1

    if match_keyword_token(text, mask, i, "begin"):
        return find_matching_begin_end(text, mask, i)

    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    n = len(text)
    while i < n:
        if not is_code_at(mask, i):
            i += 1
            continue
        ch = text[i]
        if ch == "(":
            paren_depth += 1
            i += 1
            continue
        if ch == ")":
            paren_depth = max(paren_depth - 1, 0)
            i += 1
            continue
        if ch == "[":
            bracket_depth += 1
            i += 1
            continue
        if ch == "]":
            bracket_depth = max(bracket_depth - 1, 0)
            i += 1
            continue
        if ch == "{":
            brace_depth += 1
            i += 1
            continue
        if ch == "}":
            brace_depth = max(brace_depth - 1, 0)
            i += 1
            continue
        if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0 and ch == ";":
            return i + 1
        i += 1
    return -1


def count_if_else_swap_sites(text: str, mask: list[bool]) -> int:
    count = 0
    i = 0
    n = len(text)
    while i + 1 < n:
        if not match_keyword_token(text, mask, i, "if"):
            i += 1
            continue
        j = i + 2
        while j < n and ((not is_code_at(mask, j)) or text[j].isspace()):
            j += 1
        if j >= n or not is_code_at(mask, j) or text[j] != "(":
            i += 1
            continue
        k = find_matching_paren(text, mask, j)
        if k < 0:
            i += 1
            continue
        then_start = find_next_code_nonspace(text, mask, k + 1)
        if then_start < 0:
            i += 1
            continue
        then_end = find_if_else_branch_end(text, mask, then_start)
        if then_end < 0:
            i += 1
            continue
        else_pos = find_next_code_nonspace(text, mask, then_end)
        if else_pos < 0 or not match_keyword_token(text, mask, else_pos, "else"):
            i += 1
            continue
        else_start = find_next_code_nonspace(text, mask, else_pos + 4)
        if else_start < 0:
            i += 1
            continue
        else_end = find_if_else_branch_end(text, mask, else_start)
        if else_end < 0:
            i += 1
            continue
        count += 1
        i += 1
    return count


def count_mux_swap_arms_sites(text: str, mask: list[bool]) -> int:
    count = 0
    for i, ch in enumerate(text):
        if ch != "?" or not is_code_at(mask, i):
            continue
        prev_sig = find_prev_code_nonspace(text, mask, i)
        next_sig = find_next_code_nonspace(text, mask, i + 1)
        if prev_sig < 0 or next_sig < 0:
            continue
        if not is_operand_end_char(text[prev_sig]) or not is_operand_start_char(text[next_sig]):
            continue
        stmt_start = find_statement_start(text, mask, i)
        if statement_has_assignment_disqualifier(text, mask, stmt_start, i, include_assign=False):
            continue
        if statement_looks_like_typed_declaration(text, mask, stmt_start, i):
            continue
        if not statement_has_assignment_before(text, mask, stmt_start, i):
            continue
        if find_matching_ternary_colon(text, mask, i) < 0:
            continue
        count += 1
    return count


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


def count_keyword_token(text: str, token: str, mask: list[bool]) -> int:
    if not token:
        return 0
    count = 0
    i = 0
    n = len(text)
    tlen = len(token)
    while i + tlen <= n:
        if not is_code_span(mask, i, i + tlen) or not text.startswith(token, i):
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(mask, i - 1) else ""
        nxt = text[i + tlen] if i + tlen < n and is_code_at(mask, i + tlen) else ""
        prev_boundary = (not prev) or (not is_identifier_body(prev))
        next_boundary = (not nxt) or (not is_identifier_body(nxt))
        if prev_boundary and next_boundary:
            count += 1
        i += 1
    return count


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
    if op == "IF_COND_NEGATE":
        return count_if_cond_negate_sites(design_text, mask)
    if op == "IF_ELSE_SWAP_ARMS":
        return count_if_else_swap_sites(design_text, mask)
    if op == "MUX_SWAP_ARMS":
        return count_mux_swap_arms_sites(design_text, mask)
    if op == "POSEDGE_TO_NEGEDGE":
        return count_keyword_token(design_text, "posedge", mask)
    if op == "NEGEDGE_TO_POSEDGE":
        return count_keyword_token(design_text, "negedge", mask)
    if op == "LE_TO_LT":
        return count_relational_comparator_token(design_text, "<=", mask)
    if op == "GE_TO_GT":
        return count_relational_comparator_token(design_text, ">=", mask)
    pattern = OP_PATTERNS[op]
    if op in (
        "EQ_TO_NEQ",
        "NEQ_TO_EQ",
        "AND_TO_OR",
        "OR_TO_AND",
        "XOR_TO_OR",
        "INC_TO_DEC",
        "DEC_TO_INC",
    ):
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

#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-d', '--design', required=True)
args = parser.parse_args()

input_line = Path(args.input).read_text(encoding='utf-8').strip()
parts = input_line.split(maxsplit=1)
label = parts[1] if len(parts) > 1 else ''
text = Path(args.design).read_text(encoding='utf-8')

op = label
if op.startswith('NATIVE_'):
    op = op[len('NATIVE_'):]

site_index = 1
if '@' in op:
    op_base, site_suffix = op.rsplit('@', maxsplit=1)
    if site_suffix.isdigit() and int(site_suffix) > 0:
        op = op_base
        site_index = int(site_suffix)

changed = False
code_mask = []


def build_code_mask(source: str):
    mask = [True] * len(source)
    state = "normal"
    escape = False
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if state == "normal":
            if ch == "/" and i + 1 < n and source[i + 1] == "/":
                mask[i] = False
                mask[i + 1] = False
                i += 2
                state = "line_comment"
                continue
            if ch == "/" and i + 1 < n and source[i + 1] == "*":
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
            if ch == "*" and i + 1 < n and source[i + 1] == "/":
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


def is_code_at(pos: int) -> bool:
    return 0 <= pos < len(code_mask) and code_mask[pos]


def is_code_span(start: int, end: int) -> bool:
    if start < 0 or end > len(code_mask) or start >= end:
        return False
    return all(code_mask[i] for i in range(start, end))


def replace_nth(pattern, repl, nth: int):
    global text
    if nth < 1:
        return False
    matches = [m for m in re.finditer(pattern, text) if is_code_span(m.start(), m.end())]
    if len(matches) < nth:
        return False
    match = matches[nth - 1]
    replacement = repl(match) if callable(repl) else repl
    text = text[:match.start()] + replacement + text[match.end():]
    return True


def flip_const01(token: str) -> str:
    if token == "1'b0":
        return "1'b1"
    if token == "1'd0":
        return "1'd1"
    if token == "1'h0":
        return "1'h1"
    if token == "'0":
        return "'1"
    if token == "1'b1":
        return "1'b0"
    if token == "1'd1":
        return "1'd0"
    if token == "1'h1":
        return "1'h0"
    if token == "'1":
        return "'0"
    return token


def find_comparator_token(token: str, nth: int) -> int:
    if nth < 1:
        return -1
    if token not in ("==", "!=", "===", "!=="):
        return -1
    seen = 0
    i = 0
    n = len(text)
    tlen = len(token)
    while i + tlen <= n:
        if not is_code_span(i, i + tlen):
            i += 1
            continue
        if not text.startswith(token, i):
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + tlen] if i + tlen < n and is_code_at(i + tlen) else ""
        if token == "==":
            if prev in ("=", "!", "<", ">") or nxt == "=":
                i += 1
                continue
        elif token == "!=":
            if nxt == "=":
                i += 1
                continue
        elif token in ("===", "!=="):
            if prev == "=" or nxt == "=":
                i += 1
                continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_relational_comparator_token(token: str, nth: int) -> int:
    if nth < 1:
        return -1
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    saw_plain_assign = False
    seen = 0
    i = 0
    n = len(text)
    while i + 1 < n:
        if not is_code_at(i):
            i += 1
            continue
        ch = text[i]
        nxt = text[i + 1] if is_code_at(i + 1) else ''

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
            prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ''
            if prev not in ('=', '!', '<', '>') and nxt != '=':
                saw_plain_assign = True

        if is_code_span(i, i + len(token)) and text.startswith(token, i):
            prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ''
            if token == '<=' and prev == '<':
                i += 1
                continue
            if token == '>=' and prev == '>':
                i += 1
                continue
            if paren_depth > 0 or bracket_depth > 0 or brace_depth > 0 or saw_plain_assign:
                seen += 1
                if seen == nth:
                    return i
        i += 1
    return -1


def find_statement_start(pos: int) -> int:
    i = pos
    while i > 0:
        i -= 1
        if not is_code_at(i):
            continue
        if text[i] in (";", "\n", "{", "}"):
            return i + 1
    return 0


def statement_has_plain_assign_before(stmt_start: int, pos: int) -> bool:
    i = stmt_start
    n = min(pos, len(text))
    while i < n:
        if not is_code_at(i) or text[i] != "=":
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 1] if i + 1 < len(text) and is_code_at(i + 1) else ""
        if prev in ("=", "!", "<", ">") or nxt in ("=", ">"):
            i += 1
            continue
        return True
    return False


def statement_has_assignment_before(stmt_start: int, pos: int) -> bool:
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    i = max(0, stmt_start)
    end = min(pos, len(text))
    while i < end:
        if not is_code_at(i):
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
            prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
            nxt = text[i + 1] if i + 1 < len(text) and is_code_at(i + 1) else ""
            if prev in ("=", "!", "<", ">") or nxt in ("=", ">"):
                i += 1
                continue
            return True

        if is_code_span(i, i + 2) and text.startswith("<=", i):
            prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
            nxt = text[i + 2] if i + 2 < len(text) and is_code_at(i + 2) else ""
            if prev in ("<", "=", "!", ">") or nxt in ("=", ">"):
                i += 1
                continue
            return True

        i += 1
    return False


def statement_has_assignment_disqualifier(stmt_start: int, pos: int, include_assign: bool = True) -> bool:
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
        if not is_code_at(i) or not (text[i].isalpha() or text[i] == "_"):
            i += 1
            continue
        start = i
        i += 1
        while i < end and is_code_at(i) and (
            text[i].isalnum() or text[i] in ("_", "$")
        ):
            i += 1
        token = text[start:i].lower()
        if token in disqualifiers:
            return True
    return False


def _skip_code_ws(i: int, end: int) -> int:
    while i < end:
        if not is_code_at(i):
            i += 1
            continue
        if text[i].isspace():
            i += 1
            continue
        break
    return i


def _parse_identifier_token(i: int, end: int):
    if i >= end or not is_code_at(i):
        return (-1, -1)
    ch = text[i]
    if not (ch.isalpha() or ch == "_"):
        return (-1, -1)
    start = i
    i += 1
    while i < end and is_code_at(i) and (text[i].isalnum() or text[i] in ("_", "$")):
        i += 1
    return (start, i)


def _skip_balanced(i: int, end: int, open_ch: str, close_ch: str) -> int:
    if i >= end or not is_code_at(i) or text[i] != open_ch:
        return i
    depth = 0
    while i < end:
        if not is_code_at(i):
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


def statement_looks_like_typed_declaration(stmt_start: int, pos: int) -> bool:
    end = min(pos, len(text))
    i = _skip_code_ws(max(0, stmt_start), end)
    first_start, first_end = _parse_identifier_token(i, end)
    if first_start < 0:
        return False
    first_token = text[first_start:first_end].lower()
    if first_token in {"assign", "if", "for", "while", "case", "foreach", "return", "begin", "end"}:
        return False
    i = first_end

    while True:
        i = _skip_code_ws(i, end)
        if i + 1 < end and is_code_span(i, i + 2) and text.startswith("::", i):
            i = _skip_code_ws(i + 2, end)
            _, i = _parse_identifier_token(i, end)
            if i < 0:
                return False
            continue
        if i < end and is_code_at(i) and text[i] == "#":
            i = _skip_code_ws(i + 1, end)
            i = _skip_balanced(i, end, "(", ")")
            continue
        if i < end and is_code_at(i) and text[i] == "[":
            i = _skip_balanced(i, end, "[", "]")
            continue
        break

    i = _skip_code_ws(i, end)
    second_start, _ = _parse_identifier_token(i, end)
    if second_start < 0:
        return False
    prev_sig = find_prev_code_nonspace(second_start)
    if prev_sig >= 0 and text[prev_sig] == ".":
        return False
    return True


def is_identifier_body(ch: str) -> bool:
    return ch.isalnum() or ch in ("_", "$")


def find_matching_paren(open_pos: int) -> int:
    if open_pos < 0 or open_pos >= len(text):
        return -1
    if not is_code_at(open_pos) or text[open_pos] != "(":
        return -1
    depth = 0
    i = open_pos
    n = len(text)
    while i < n:
        if not is_code_at(i):
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


def find_if_cond_token(nth: int):
    if nth < 1:
        return (-1, -1, -1)
    seen = 0
    i = 0
    n = len(text)
    while i + 1 < n:
        if not is_code_span(i, i + 2) or not text.startswith("if", i):
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 2] if i + 2 < n and is_code_at(i + 2) else ""
        if (prev and is_identifier_body(prev)) or (nxt and is_identifier_body(nxt)):
            i += 1
            continue
        j = i + 2
        while j < n and ((not is_code_at(j)) or text[j].isspace()):
            j += 1
        if j >= n or not is_code_at(j) or text[j] != "(":
            i += 1
            continue
        k = find_matching_paren(j)
        if k < 0:
            i += 1
            continue
        seen += 1
        if seen == nth:
            return (i, j, k)
        i += 1
        continue
    return (-1, -1, -1)


def match_keyword_token(pos: int, keyword: str) -> bool:
    if pos < 0 or not keyword:
        return False
    klen = len(keyword)
    if pos + klen > len(text):
        return False
    if not is_code_span(pos, pos + klen) or not text.startswith(keyword, pos):
        return False
    prev = text[pos - 1] if pos > 0 and is_code_at(pos - 1) else ""
    nxt = text[pos + klen] if pos + klen < len(text) and is_code_at(pos + klen) else ""
    prev_boundary = (not prev) or (not is_identifier_body(prev))
    next_boundary = (not nxt) or (not is_identifier_body(nxt))
    return prev_boundary and next_boundary


def find_matching_begin_end(begin_pos: int) -> int:
    if not match_keyword_token(begin_pos, "begin"):
        return -1
    depth = 0
    i = begin_pos
    n = len(text)
    while i < n:
        if not is_code_at(i):
            i += 1
            continue
        if match_keyword_token(i, "begin"):
            depth += 1
            i += len("begin")
            continue
        if match_keyword_token(i, "end"):
            depth -= 1
            i += len("end")
            if depth == 0:
                return i
            if depth < 0:
                return -1
            continue
        i += 1
    return -1


def find_if_else_branch_end(branch_start: int) -> int:
    i = find_next_code_nonspace(branch_start)
    if i < 0:
        return -1

    # Skip ambiguous dangling-else forms (`else if ...`) to keep rewrites
    # structurally deterministic and semantically explicit.
    if match_keyword_token(i, "if"):
        return -1

    if match_keyword_token(i, "begin"):
        return find_matching_begin_end(i)

    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    n = len(text)
    while i < n:
        if not is_code_at(i):
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


def find_if_else_swap_token(nth: int):
    if nth < 1:
        return (-1, -1, -1, -1, -1, -1)
    seen = 0
    i = 0
    n = len(text)
    while i + 1 < n:
        if not match_keyword_token(i, "if"):
            i += 1
            continue
        j = i + 2
        while j < n and ((not is_code_at(j)) or text[j].isspace()):
            j += 1
        if j >= n or not is_code_at(j) or text[j] != "(":
            i += 1
            continue
        k = find_matching_paren(j)
        if k < 0:
            i += 1
            continue
        then_start = find_next_code_nonspace(k + 1)
        if then_start < 0:
            i += 1
            continue
        then_end = find_if_else_branch_end(then_start)
        if then_end < 0:
            i += 1
            continue
        else_pos = find_next_code_nonspace(then_end)
        if else_pos < 0 or not match_keyword_token(else_pos, "else"):
            i += 1
            continue
        else_start = find_next_code_nonspace(else_pos + 4)
        if else_start < 0:
            i += 1
            continue
        else_end = find_if_else_branch_end(else_start)
        if else_end < 0:
            i += 1
            continue
        seen += 1
        if seen == nth:
            return (i, then_start, then_end, else_pos, else_start, else_end)
        i += 1
    return (-1, -1, -1, -1, -1, -1)


def find_keyword_token(keyword: str, nth: int) -> int:
    if nth < 1 or not keyword:
        return -1
    seen = 0
    i = 0
    n = len(text)
    klen = len(keyword)
    while i + klen <= n:
        if not is_code_span(i, i + klen) or not text.startswith(keyword, i):
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + klen] if i + klen < n and is_code_at(i + klen) else ""
        prev_boundary = (not prev) or (not is_identifier_body(prev))
        next_boundary = (not nxt) or (not is_identifier_body(nxt))
        if not prev_boundary or not next_boundary:
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_matching_ternary_colon(question_pos: int) -> int:
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    nested_ternary = 0
    i = question_pos + 1
    n = len(text)
    while i < n:
        if not is_code_at(i):
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


def find_ternary_end_delimiter(colon_pos: int) -> int:
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    nested_ternary = 0
    i = colon_pos + 1
    n = len(text)
    while i < n:
        if not is_code_at(i):
            i += 1
            continue
        ch = text[i]
        if ch == "(":
            paren_depth += 1
            i += 1
            continue
        if ch == ")":
            if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0 and nested_ternary == 0:
                return i
            paren_depth = max(paren_depth - 1, 0)
            i += 1
            continue
        if ch == "[":
            bracket_depth += 1
            i += 1
            continue
        if ch == "]":
            if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0 and nested_ternary == 0:
                return i
            bracket_depth = max(bracket_depth - 1, 0)
            i += 1
            continue
        if ch == "{":
            brace_depth += 1
            i += 1
            continue
        if ch == "}":
            if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0 and nested_ternary == 0:
                return i
            brace_depth = max(brace_depth - 1, 0)
            i += 1
            continue

        if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
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
            if nested_ternary == 0 and ch in (";", ","):
                return i
        i += 1
    return n


def find_ternary_mux_token(nth: int):
    if nth < 1:
        return (-1, -1, -1)
    seen = 0
    i = 0
    n = len(text)
    while i < n:
        if not is_code_at(i) or text[i] != "?":
            i += 1
            continue
        prev_sig = find_prev_code_nonspace(i)
        next_sig = find_next_code_nonspace(i + 1)
        if prev_sig < 0 or next_sig < 0:
            i += 1
            continue
        if not is_operand_end_char(text[prev_sig]) or not is_operand_start_char(text[next_sig]):
            i += 1
            continue
        stmt_start = find_statement_start(i)
        if statement_has_assignment_disqualifier(stmt_start, i, include_assign=False):
            i += 1
            continue
        if statement_looks_like_typed_declaration(stmt_start, i):
            i += 1
            continue
        if not statement_has_assignment_before(stmt_start, i):
            i += 1
            continue
        colon_idx = find_matching_ternary_colon(i)
        if colon_idx < 0:
            i += 1
            continue
        true_start = find_next_code_nonspace(i + 1)
        true_end = find_prev_code_nonspace(colon_idx)
        false_start = find_next_code_nonspace(colon_idx + 1)
        if true_start < 0 or true_end < 0 or false_start < 0 or true_start > true_end:
            i += 1
            continue
        end_delim = find_ternary_end_delimiter(colon_idx)
        false_end = find_prev_code_nonspace(end_delim)
        if false_end < false_start:
            i += 1
            continue
        seen += 1
        if seen == nth:
            return (i, colon_idx, end_delim)
        i += 1
        continue
    return (-1, -1, -1)


def find_procedural_blocking_assign_token(nth: int) -> int:
    if nth < 1:
        return -1
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    seen = 0
    i = 0
    n = len(text)
    while i < n:
        if not is_code_at(i):
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
        if ch != "=":
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 1] if i + 1 < n and is_code_at(i + 1) else ""
        if prev in ("=", "!", "<", ">") or nxt in ("=", ">"):
            i += 1
            continue
        if paren_depth > 0 or bracket_depth > 0 or brace_depth > 0:
            i += 1
            continue
        prev_sig = find_prev_code_nonspace(i)
        next_sig = find_next_code_nonspace(i + 1)
        if prev_sig < 0 or next_sig < 0:
            i += 1
            continue
        if not is_operand_end_char(text[prev_sig]) or not is_operand_start_char(
            text[next_sig]
        ):
            i += 1
            continue
        stmt_start = find_statement_start(i)
        if statement_has_assignment_disqualifier(stmt_start, i):
            i += 1
            continue
        if statement_looks_like_typed_declaration(stmt_start, i):
            i += 1
            continue
        if statement_has_plain_assign_before(stmt_start, i):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_procedural_nonblocking_assign_token(nth: int) -> int:
    if nth < 1:
        return -1
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    seen = 0
    i = 0
    n = len(text)
    while i + 1 < n:
        if not is_code_at(i):
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
        if not is_code_span(i, i + 2) or not text.startswith("<=", i):
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 2] if i + 2 < n and is_code_at(i + 2) else ""
        if prev in ("<", "=", "!", ">") or nxt in ("=", ">"):
            i += 1
            continue
        if paren_depth > 0 or bracket_depth > 0 or brace_depth > 0:
            i += 1
            continue
        prev_sig = find_prev_code_nonspace(i)
        next_sig = find_next_code_nonspace(i + 2)
        if prev_sig < 0 or next_sig < 0:
            i += 1
            continue
        if not is_operand_end_char(text[prev_sig]) or not is_operand_start_char(
            text[next_sig]
        ):
            i += 1
            continue
        stmt_start = find_statement_start(i)
        if statement_has_assignment_disqualifier(stmt_start, i):
            i += 1
            continue
        if statement_looks_like_typed_declaration(stmt_start, i):
            i += 1
            continue
        if statement_has_plain_assign_before(stmt_start, i):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_standalone_compare_token(token: str, nth: int) -> int:
    if nth < 1:
        return -1
    if token not in ("<", ">"):
        return -1
    seen = 0
    i = 0
    n = len(text)
    while i < n:
        if not is_code_at(i):
            i += 1
            continue
        if text[i] != token:
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 1] if i + 1 < n and is_code_at(i + 1) else ""
        if prev in ("<", ">", "=", "!") or nxt in ("<", ">", "=", "!"):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_prev_code_nonspace(pos: int) -> int:
    i = pos
    while i > 0:
        i -= 1
        if not is_code_at(i):
            continue
        if text[i].isspace():
            continue
        return i
    return -1


def find_next_code_nonspace(pos: int) -> int:
    i = pos
    n = len(text)
    while i < n:
        if not is_code_at(i):
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


def is_unary_operator_context(ch: str) -> bool:
    return ch in ("(", "[", "{", ":", ";", ",", "?", "=", "+", "-", "*", "/", "%", "&", "|", "^", "!", "~", "<", ">")


def find_binary_arithmetic_token(token: str, nth: int) -> int:
    if nth < 1:
        return -1
    if token not in ("+", "-"):
        return -1
    seen = 0
    i = 0
    n = len(text)
    bracket_depth = 0
    while i < n:
        if not is_code_at(i):
            i += 1
            continue
        ch = text[i]
        if ch == "[":
            bracket_depth += 1
            i += 1
            continue
        if ch == "]":
            bracket_depth = max(bracket_depth - 1, 0)
            i += 1
            continue
        if ch != token:
            i += 1
            continue
        if bracket_depth > 0:
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 1] if i + 1 < n and is_code_at(i + 1) else ""
        if prev == token or nxt == token:
            i += 1
            continue
        if prev == "=" or nxt == "=":
            i += 1
            continue
        if token == "-" and nxt == ">":
            i += 1
            continue
        prev_sig = find_prev_code_nonspace(i)
        next_sig = find_next_code_nonspace(i + 1)
        if prev_sig < 0 or next_sig < 0:
            i += 1
            continue
        if not is_operand_end_char(text[prev_sig]) or not is_operand_start_char(
            text[next_sig]
        ):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_binary_muldiv_token(token: str, nth: int) -> int:
    if nth < 1:
        return -1
    if token not in ("*", "/"):
        return -1
    seen = 0
    i = 0
    n = len(text)
    bracket_depth = 0
    while i < n:
        if not is_code_at(i):
            i += 1
            continue
        ch = text[i]
        if ch == "[":
            bracket_depth += 1
            i += 1
            continue
        if ch == "]":
            bracket_depth = max(bracket_depth - 1, 0)
            i += 1
            continue
        if ch != token:
            i += 1
            continue
        if bracket_depth > 0:
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 1] if i + 1 < n and is_code_at(i + 1) else ""
        if token == "*":
            if prev == "*" or nxt == "*":
                i += 1
                continue
            if nxt == "=":
                i += 1
                continue
            if prev == "(" and nxt == ")":
                i += 1
                continue
        else:
            if prev == "/" or nxt == "/":
                i += 1
                continue
            if nxt == "=":
                i += 1
                continue
        prev_sig = find_prev_code_nonspace(i)
        next_sig = find_next_code_nonspace(i + 1)
        if prev_sig < 0 or next_sig < 0:
            i += 1
            continue
        if not is_operand_end_char(text[prev_sig]) or not is_operand_start_char(
            text[next_sig]
        ):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_unary_minus_token(nth: int) -> int:
    if nth < 1:
        return -1
    seen = 0
    i = 0
    n = len(text)

    def is_unary_context(ch: str) -> bool:
        return ch in "([{:,;?=+-*/%&|^!~<>"

    while i < n:
        if not is_code_at(i) or text[i] != "-":
            i += 1
            continue
        prev_immediate = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        next_immediate = text[i + 1] if i + 1 < n and is_code_at(i + 1) else ""
        if prev_immediate == "-" or next_immediate == "-":
            i += 1
            continue
        if next_immediate == ">":
            i += 1
            continue
        prev_sig = find_prev_code_nonspace(i)
        if prev_sig >= 0 and not is_unary_context(text[prev_sig]):
            i += 1
            continue
        next_sig = find_next_code_nonspace(i + 1)
        if next_sig < 0:
            i += 1
            continue
        if not is_operand_start_char(text[next_sig]):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_binary_shift_token(token: str, nth: int) -> int:
    if nth < 1:
        return -1
    if token not in ("<<", ">>"):
        return -1
    marker = token[0]
    seen = 0
    i = 0
    n = len(text)
    while i + 1 < n:
        if not is_code_span(i, i + 2):
            i += 1
            continue
        if not text.startswith(token, i):
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 2] if i + 2 < n and is_code_at(i + 2) else ""
        if prev == marker or nxt == marker:
            i += 1
            continue
        if nxt == "=":
            i += 1
            continue
        prev_sig = find_prev_code_nonspace(i)
        next_sig = find_next_code_nonspace(i + 2)
        if prev_sig < 0 or next_sig < 0:
            i += 1
            continue
        if not is_operand_end_char(text[prev_sig]) or not is_operand_start_char(
            text[next_sig]
        ):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_binary_ashr_token(nth: int) -> int:
    if nth < 1:
        return -1
    seen = 0
    i = 0
    n = len(text)
    while i + 2 < n:
        if not is_code_span(i, i + 3):
            i += 1
            continue
        if not text.startswith(">>>", i):
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 3] if i + 3 < n and is_code_at(i + 3) else ""
        if prev == ">" or nxt == ">":
            i += 1
            continue
        if nxt == "=":
            i += 1
            continue
        prev_sig = find_prev_code_nonspace(i)
        next_sig = find_next_code_nonspace(i + 3)
        if prev_sig < 0 or next_sig < 0:
            i += 1
            continue
        if not is_operand_end_char(text[prev_sig]) or not is_operand_start_char(
            text[next_sig]
        ):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_logical_token(token: str, nth: int) -> int:
    if nth < 1:
        return -1
    if token not in ("&&", "||"):
        return -1
    marker = token[0]
    seen = 0
    i = 0
    n = len(text)
    while i + 1 < n:
        if not is_code_span(i, i + 2):
            i += 1
            continue
        if not text.startswith(token, i):
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 2] if i + 2 < n and is_code_at(i + 2) else ""
        if prev == marker or nxt == marker:
            i += 1
            continue
        prev_sig = find_prev_code_nonspace(i)
        next_sig = find_next_code_nonspace(i + 2)
        if prev_sig < 0 or next_sig < 0:
            i += 1
            continue
        if not is_operand_end_char(text[prev_sig]) or not is_operand_start_char(
            text[next_sig]
        ):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_binary_xor_token(nth: int) -> int:
    if nth < 1:
        return -1
    seen = 0
    i = 0
    n = len(text)
    while i < n:
        if not is_code_at(i):
            i += 1
            continue
        if text[i] != "^":
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 1] if i + 1 < n and is_code_at(i + 1) else ""
        if prev == "=" or nxt == "=":
            i += 1
            continue
        if prev == "~" or nxt == "~":
            i += 1
            continue
        prev_sig = find_prev_code_nonspace(i)
        next_sig = find_next_code_nonspace(i + 1)
        if prev_sig < 0 or next_sig < 0:
            i += 1
            continue
        if not is_operand_end_char(text[prev_sig]) or not is_operand_start_char(
            text[next_sig]
        ):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_binary_xnor_token(nth: int) -> int:
    if nth < 1:
        return -1
    seen = 0
    i = 0
    n = len(text)
    while i + 1 < n:
        if not is_code_span(i, i + 2):
            i += 1
            continue
        if text.startswith("^~", i) or text.startswith("~^", i):
            prev_sig = find_prev_code_nonspace(i)
            next_sig = find_next_code_nonspace(i + 2)
            if prev_sig < 0 or next_sig < 0:
                i += 1
                continue
            if not is_operand_end_char(text[prev_sig]) or not is_operand_start_char(
                text[next_sig]
            ):
                i += 1
                continue
            seen += 1
            if seen == nth:
                return i
        i += 1
    return -1


def find_binary_bitwise_token(token: str, nth: int) -> int:
    if nth < 1:
        return -1
    if token not in ("&", "|"):
        return -1
    seen = 0
    i = 0
    n = len(text)
    while i < n:
        if not is_code_at(i):
            i += 1
            continue
        if text[i] != token:
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 1] if i + 1 < n and is_code_at(i + 1) else ""
        if prev == token or nxt == token:
            i += 1
            continue
        if prev == "=" or nxt == "=":
            i += 1
            continue
        if prev == "~":
            i += 1
            continue
        prev_sig = find_prev_code_nonspace(i)
        next_sig = find_next_code_nonspace(i + 1)
        if prev_sig < 0 or next_sig < 0:
            i += 1
            continue
        if not is_operand_end_char(text[prev_sig]) or not is_operand_start_char(
            text[next_sig]
        ):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_unary_bitnot_token(nth: int) -> int:
    if nth < 1:
        return -1
    seen = 0
    i = 0
    n = len(text)
    while i < n:
        if not is_code_at(i):
            i += 1
            continue
        if text[i] != "~":
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        if prev == "^":
            i += 1
            continue
        nxt = text[i + 1] if i + 1 < n and is_code_at(i + 1) else ""
        if nxt in ("&", "|", "^", "="):
            i += 1
            continue
        j = i + 1
        while j < n and is_code_at(j) and text[j].isspace():
            j += 1
        if j >= n or not is_code_at(j):
            i += 1
            continue
        if not is_operand_start_char(text[j]):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_unary_reduction_token(token: str, nth: int) -> int:
    if nth < 1:
        return -1
    if token not in ("&", "|", "^"):
        return -1
    seen = 0
    i = 0
    n = len(text)
    while i < n:
        if not is_code_at(i):
            i += 1
            continue
        if text[i] != token:
            i += 1
            continue
        prev = text[i - 1] if i > 0 and is_code_at(i - 1) else ""
        nxt = text[i + 1] if i + 1 < n and is_code_at(i + 1) else ""
        if token != "^" and (prev == token or nxt == token):
            i += 1
            continue
        if prev == "=" or nxt == "=":
            i += 1
            continue
        if token == "^" and (prev == "~" or nxt == "~"):
            i += 1
            continue
        if prev == "~":
            i += 1
            continue
        prev_sig = find_prev_code_nonspace(i)
        if prev_sig >= 0 and not is_unary_operator_context(text[prev_sig]):
            i += 1
            continue
        next_sig = find_next_code_nonspace(i + 1)
        if next_sig < 0:
            i += 1
            continue
        if not is_operand_start_char(text[next_sig]):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_unary_reduction_xnor_token(nth: int) -> int:
    if nth < 1:
        return -1
    seen = 0
    i = 0
    n = len(text)
    while i + 1 < n:
        if not is_code_span(i, i + 2):
            i += 1
            continue
        if not (text.startswith("^~", i) or text.startswith("~^", i)):
            i += 1
            continue
        prev_sig = find_prev_code_nonspace(i)
        if prev_sig >= 0 and not is_unary_operator_context(text[prev_sig]):
            i += 1
            continue
        next_sig = find_next_code_nonspace(i + 2)
        if next_sig < 0 or not is_operand_start_char(text[next_sig]):
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


def find_cast_function_token(name: str, nth: int) -> int:
    if nth < 1:
        return -1
    if name not in ("signed", "unsigned"):
        return -1
    seen = 0
    i = 0
    n = len(text)
    name_len = len(name)
    while i < n:
        if not is_code_at(i) or text[i] != "$":
            i += 1
            continue
        if i + 1 + name_len > n:
            i += 1
            continue
        if not is_code_span(i + 1, i + 1 + name_len):
            i += 1
            continue
        if text[i + 1 : i + 1 + name_len] != name:
            i += 1
            continue
        end_name = i + 1 + name_len
        nxt = text[end_name] if end_name < n and is_code_at(end_name) else ""
        if nxt.isalnum() or nxt in ("_", "$"):
            i += 1
            continue
        j = end_name
        while j < n and is_code_at(j) and text[j].isspace():
            j += 1
        if j >= n or not is_code_at(j) or text[j] != "(":
            i += 1
            continue
        seen += 1
        if seen == nth:
            return i
        i += 1
    return -1


code_mask = build_code_mask(text)

if op == 'EQ_TO_NEQ':
    idx = find_comparator_token('==', site_index)
    if idx >= 0:
        text = text[:idx] + '!=' + text[idx + 2:]
        changed = True
elif op == 'NEQ_TO_EQ':
    idx = find_comparator_token('!=', site_index)
    if idx >= 0:
        text = text[:idx] + '==' + text[idx + 2:]
        changed = True
elif op == 'CASEEQ_TO_EQ':
    idx = find_comparator_token('===', site_index)
    if idx >= 0:
        text = text[:idx] + '==' + text[idx + 3:]
        changed = True
elif op == 'CASENEQ_TO_NEQ':
    idx = find_comparator_token('!==', site_index)
    if idx >= 0:
        text = text[:idx] + '!=' + text[idx + 3:]
        changed = True
elif op == 'EQ_TO_CASEEQ':
    idx = find_comparator_token('==', site_index)
    if idx >= 0:
        text = text[:idx] + '===' + text[idx + 2:]
        changed = True
elif op == 'NEQ_TO_CASENEQ':
    idx = find_comparator_token('!=', site_index)
    if idx >= 0:
        text = text[:idx] + '!==' + text[idx + 2:]
        changed = True
elif op == 'SIGNED_TO_UNSIGNED':
    idx = find_cast_function_token('signed', site_index)
    if idx >= 0:
        text = text[:idx] + '$unsigned' + text[idx + len('$signed') :]
        changed = True
elif op == 'UNSIGNED_TO_SIGNED':
    idx = find_cast_function_token('unsigned', site_index)
    if idx >= 0:
        text = text[:idx] + '$signed' + text[idx + len('$unsigned') :]
        changed = True
elif op == 'LT_TO_LE':
    idx = find_standalone_compare_token('<', site_index)
    if idx >= 0:
        text = text[:idx] + '<=' + text[idx + 1:]
        changed = True
elif op == 'GT_TO_GE':
    idx = find_standalone_compare_token('>', site_index)
    if idx >= 0:
        text = text[:idx] + '>=' + text[idx + 1:]
        changed = True
elif op == 'LE_TO_LT':
    idx = find_relational_comparator_token('<=', site_index)
    if idx >= 0:
        text = text[:idx] + '<' + text[idx + 2:]
        changed = True
elif op == 'GE_TO_GT':
    idx = find_relational_comparator_token('>=', site_index)
    if idx >= 0:
        text = text[:idx] + '>' + text[idx + 2:]
        changed = True
elif op == 'LT_TO_GT':
    idx = find_standalone_compare_token('<', site_index)
    if idx >= 0:
        text = text[:idx] + '>' + text[idx + 1:]
        changed = True
elif op == 'GT_TO_LT':
    idx = find_standalone_compare_token('>', site_index)
    if idx >= 0:
        text = text[:idx] + '<' + text[idx + 1:]
        changed = True
elif op == 'LE_TO_GE':
    idx = find_relational_comparator_token('<=', site_index)
    if idx >= 0:
        text = text[:idx] + '>=' + text[idx + 2:]
        changed = True
elif op == 'GE_TO_LE':
    idx = find_relational_comparator_token('>=', site_index)
    if idx >= 0:
        text = text[:idx] + '<=' + text[idx + 2:]
        changed = True
elif op == 'AND_TO_OR':
    changed = replace_nth(r'&&', '||', site_index)
elif op == 'OR_TO_AND':
    changed = replace_nth(r'\|\|', '&&', site_index)
elif op == 'LAND_TO_BAND':
    idx = find_logical_token('&&', site_index)
    if idx >= 0:
        text = text[:idx] + '&' + text[idx + 2:]
        changed = True
elif op == 'LOR_TO_BOR':
    idx = find_logical_token('||', site_index)
    if idx >= 0:
        text = text[:idx] + '|' + text[idx + 2:]
        changed = True
elif op == 'XOR_TO_OR':
    idx = find_binary_xor_token(site_index)
    if idx >= 0:
        text = text[:idx] + '|' + text[idx + 1:]
        changed = True
elif op == 'XOR_TO_XNOR':
    idx = find_binary_xor_token(site_index)
    if idx >= 0:
        text = text[:idx] + '^~' + text[idx + 1:]
        changed = True
elif op == 'XNOR_TO_XOR':
    idx = find_binary_xnor_token(site_index)
    if idx >= 0:
        text = text[:idx] + '^' + text[idx + 2:]
        changed = True
elif op == 'REDAND_TO_REDOR':
    idx = find_unary_reduction_token('&', site_index)
    if idx >= 0:
        text = text[:idx] + '|' + text[idx + 1:]
        changed = True
elif op == 'REDOR_TO_REDAND':
    idx = find_unary_reduction_token('|', site_index)
    if idx >= 0:
        text = text[:idx] + '&' + text[idx + 1:]
        changed = True
elif op == 'REDXOR_TO_REDXNOR':
    idx = find_unary_reduction_token('^', site_index)
    if idx >= 0:
        text = text[:idx] + '^~' + text[idx + 1:]
        changed = True
elif op == 'REDXNOR_TO_REDXOR':
    idx = find_unary_reduction_xnor_token(site_index)
    if idx >= 0:
        text = text[:idx] + '^' + text[idx + 2:]
        changed = True
elif op == 'BAND_TO_BOR':
    idx = find_binary_bitwise_token('&', site_index)
    if idx >= 0:
        text = text[:idx] + '|' + text[idx + 1:]
        changed = True
elif op == 'BOR_TO_BAND':
    idx = find_binary_bitwise_token('|', site_index)
    if idx >= 0:
        text = text[:idx] + '&' + text[idx + 1:]
        changed = True
elif op == 'BAND_TO_LAND':
    idx = find_binary_bitwise_token('&', site_index)
    if idx >= 0:
        text = text[:idx] + '&&' + text[idx + 1:]
        changed = True
elif op == 'BOR_TO_LOR':
    idx = find_binary_bitwise_token('|', site_index)
    if idx >= 0:
        text = text[:idx] + '||' + text[idx + 1:]
        changed = True
elif op == 'BA_TO_NBA':
    idx = find_procedural_blocking_assign_token(site_index)
    if idx >= 0:
        text = text[:idx] + '<=' + text[idx + 1:]
        changed = True
elif op == 'NBA_TO_BA':
    idx = find_procedural_nonblocking_assign_token(site_index)
    if idx >= 0:
        text = text[:idx] + '=' + text[idx + 2:]
        changed = True
elif op == 'POSEDGE_TO_NEGEDGE':
    idx = find_keyword_token('posedge', site_index)
    if idx >= 0:
        text = text[:idx] + 'negedge' + text[idx + len('posedge'):]
        changed = True
elif op == 'NEGEDGE_TO_POSEDGE':
    idx = find_keyword_token('negedge', site_index)
    if idx >= 0:
        text = text[:idx] + 'posedge' + text[idx + len('negedge'):]
        changed = True
elif op == 'MUX_SWAP_ARMS':
    q_idx, colon_idx, end_delim = find_ternary_mux_token(site_index)
    if q_idx >= 0 and colon_idx >= 0 and end_delim >= 0:
        true_start = find_next_code_nonspace(q_idx + 1)
        true_end = find_prev_code_nonspace(colon_idx)
        false_start = find_next_code_nonspace(colon_idx + 1)
        false_end = find_prev_code_nonspace(end_delim)
        if true_start >= 0 and true_end >= true_start and false_start >= 0 and false_end >= false_start:
            lhs = text[q_idx + 1:true_start]
            true_expr = text[true_start:true_end + 1]
            true_to_colon_ws = text[true_end + 1:colon_idx]
            colon_to_false_ws = text[colon_idx + 1:false_start]
            false_expr = text[false_start:false_end + 1]
            false_suffix_ws = text[false_end + 1:end_delim]
            swapped = (
                text[:q_idx + 1]
                + lhs
                + false_expr
                + true_to_colon_ws
                + ':'
                + colon_to_false_ws
                + true_expr
                + false_suffix_ws
                + text[end_delim:]
            )
            text = swapped
            changed = True
elif op == 'IF_COND_NEGATE':
    _, cond_open, cond_close = find_if_cond_token(site_index)
    if cond_open >= 0 and cond_close >= cond_open:
        cond_expr = text[cond_open + 1:cond_close]
        text = text[:cond_open + 1] + '!(' + cond_expr + ')' + text[cond_close:]
        changed = True
elif op == 'IF_ELSE_SWAP_ARMS':
    _, then_start, then_end, else_pos, else_start, else_end = find_if_else_swap_token(site_index)
    if then_start >= 0 and then_end >= then_start and else_pos >= 0 and else_start >= 0 and else_end >= else_start:
        then_arm = text[then_start:then_end]
        between_arms = text[then_end:else_pos]
        else_header = text[else_pos:else_start]
        else_arm = text[else_start:else_end]
        text = (
            text[:then_start]
            + else_arm
            + between_arms
            + else_header
            + then_arm
            + text[else_end:]
        )
        changed = True
elif op == 'UNARY_NOT_DROP':
    changed = replace_nth(r'!\s*(?=[A-Za-z_(])', '', site_index)
elif op == 'UNARY_BNOT_DROP':
    idx = find_unary_bitnot_token(site_index)
    if idx >= 0:
        end = idx + 1
        while end < len(text) and is_code_at(end) and text[end].isspace():
            end += 1
        text = text[:idx] + text[end:]
        changed = True
elif op == 'CONST0_TO_1':
    changed = replace_nth(
        r"1'b0|1'd0|1'h0|'0",
        lambda m: flip_const01(m.group(0)),
        site_index,
    )
elif op == 'CONST1_TO_0':
    changed = replace_nth(
        r"1'b1|1'd1|1'h1|'1",
        lambda m: flip_const01(m.group(0)),
        site_index,
    )
elif op == 'ADD_TO_SUB':
    idx = find_binary_arithmetic_token('+', site_index)
    if idx >= 0:
        text = text[:idx] + '-' + text[idx + 1:]
        changed = True
elif op == 'SUB_TO_ADD':
    idx = find_binary_arithmetic_token('-', site_index)
    if idx >= 0:
        text = text[:idx] + '+' + text[idx + 1:]
        changed = True
elif op == 'MUL_TO_ADD':
    idx = find_binary_muldiv_token('*', site_index)
    if idx >= 0:
        text = text[:idx] + '+' + text[idx + 1:]
        changed = True
elif op == 'ADD_TO_MUL':
    idx = find_binary_arithmetic_token('+', site_index)
    if idx >= 0:
        text = text[:idx] + '*' + text[idx + 1:]
        changed = True
elif op == 'DIV_TO_MUL':
    idx = find_binary_muldiv_token('/', site_index)
    if idx >= 0:
        text = text[:idx] + '*' + text[idx + 1:]
        changed = True
elif op == 'MUL_TO_DIV':
    idx = find_binary_muldiv_token('*', site_index)
    if idx >= 0:
        text = text[:idx] + '/' + text[idx + 1:]
        changed = True
elif op == 'UNARY_MINUS_DROP':
    idx = find_unary_minus_token(site_index)
    if idx >= 0:
        end = idx + 1
        while end < len(text) and is_code_at(end) and text[end].isspace():
            end += 1
        text = text[:idx] + text[end:]
        changed = True
elif op == 'INC_TO_DEC':
    changed = replace_nth(r"\+\+", "--", site_index)
elif op == 'DEC_TO_INC':
    changed = replace_nth(r"--", "++", site_index)
elif op == 'SHL_TO_SHR':
    idx = find_binary_shift_token('<<', site_index)
    if idx >= 0:
        text = text[:idx] + '>>' + text[idx + 2:]
        changed = True
elif op == 'SHR_TO_SHL':
    idx = find_binary_shift_token('>>', site_index)
    if idx >= 0:
        text = text[:idx] + '<<' + text[idx + 2:]
        changed = True
elif op == 'SHR_TO_ASHR':
    idx = find_binary_shift_token('>>', site_index)
    if idx >= 0:
        text = text[:idx] + '>>>' + text[idx + 2:]
        changed = True
elif op == 'ASHR_TO_SHR':
    idx = find_binary_ashr_token(site_index)
    if idx >= 0:
        text = text[:idx] + '>>' + text[idx + 3:]
        changed = True

if not changed:
    text += f"\n// native_mutation_noop_fallback {label}\n"
    marker_path = os.environ.get("CIRCT_MUT_NATIVE_NOOP_FALLBACK_MARKER", "")
    if marker_path:
        with open(marker_path, "a", encoding="utf-8") as marker:
            marker.write(f"{label}\n")

Path(args.output).write_text(text, encoding='utf-8')

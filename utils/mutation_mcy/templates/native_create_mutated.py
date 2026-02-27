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
elif op == 'LT_TO_LE':
    changed = replace_nth(r'(?<![<>=!])<(?![<>=])', '<=', site_index)
elif op == 'GT_TO_GE':
    changed = replace_nth(r'(?<![<>=!])>(?![<>=])', '>=', site_index)
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
elif op == 'AND_TO_OR':
    changed = replace_nth(r'&&', '||', site_index)
elif op == 'OR_TO_AND':
    changed = replace_nth(r'\|\|', '&&', site_index)
elif op == 'XOR_TO_OR':
    idx = find_binary_xor_token(site_index)
    if idx >= 0:
        text = text[:idx] + '|' + text[idx + 1:]
        changed = True
elif op == 'UNARY_NOT_DROP':
    changed = replace_nth(r'!\s*(?=[A-Za-z_(])', '', site_index)
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

if not changed:
    text += f"\n// native_mutation_noop_fallback {label}\n"
    marker_path = os.environ.get("CIRCT_MUT_NATIVE_NOOP_FALLBACK_MARKER", "")
    if marker_path:
        with open(marker_path, "a", encoding="utf-8") as marker:
            marker.write(f"{label}\n")

Path(args.output).write_text(text, encoding='utf-8')

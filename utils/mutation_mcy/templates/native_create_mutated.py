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

changed = False


def replace_once(pattern, repl):
    global changed
    new_text, count = re.subn(pattern, repl, text, count=1)
    return new_text, count


if op == 'EQ_TO_NEQ':
    text, count = replace_once(r'==', '!=')
    changed = count > 0
elif op == 'NEQ_TO_EQ':
    text, count = replace_once(r'!=', '==')
    changed = count > 0
elif op == 'LT_TO_LE':
    text, count = replace_once(r'(?<![<>=!])<(?![<>=])', '<=')
    changed = count > 0
elif op == 'GT_TO_GE':
    text, count = replace_once(r'(?<![<>=!])>(?![<>=])', '>=')
    changed = count > 0
elif op == 'LE_TO_LT':
    text, count = replace_once(r'<=', '<')
    changed = count > 0
elif op == 'GE_TO_GT':
    text, count = replace_once(r'>=', '>')
    changed = count > 0
elif op == 'AND_TO_OR':
    text, count = replace_once(r'&&', '||')
    changed = count > 0
elif op == 'OR_TO_AND':
    text, count = replace_once(r'\|\|', '&&')
    changed = count > 0
elif op == 'XOR_TO_OR':
    text, count = replace_once(r'\^', '|')
    changed = count > 0
elif op == 'UNARY_NOT_DROP':
    text, count = replace_once(r'!\s*(?=[A-Za-z_(])', '')
    changed = count > 0
elif op == 'CONST0_TO_1':
    text, count = replace_once(r"1'b0", "1'b1")
    if count == 0:
        text, count = replace_once(r"1'd0", "1'd1")
    changed = count > 0
elif op == 'CONST1_TO_0':
    text, count = replace_once(r"1'b1", "1'b0")
    if count == 0:
        text, count = replace_once(r"1'd1", "1'd0")
    changed = count > 0

if not changed:
    m = re.search(r'assign\s+([^=]+?)\s*=\s*(.+?);', text, re.S)
    if m:
        lhs = m.group(1).strip()
        rhs = m.group(2).strip()
        repl = f'assign {lhs} = ~({rhs});'
        text = text[:m.start()] + repl + text[m.end():]
        changed = True

if not changed:
    text += f"\n// native_mutation_noop_fallback {label}\n"
    marker_path = os.environ.get("CIRCT_MUT_NATIVE_NOOP_FALLBACK_MARKER", "")
    if marker_path:
        with open(marker_path, "a", encoding="utf-8") as marker:
            marker.write(f"{label}\n")

Path(args.output).write_text(text, encoding='utf-8')

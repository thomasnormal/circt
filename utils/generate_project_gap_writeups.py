#!/usr/bin/env python3
"""Generate paragraph-style writeups for project gap markers.

Input format is expected to match ripgrep output:
  path:line:matched text
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sys


ENTRY_RE = re.compile(r"^(.*?):(\d+):(.*)$")
TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|TBD|NYI|WIP)\b", re.IGNORECASE)
UNSUPPORTED_RE = re.compile(
    r"\b(unsupported|unimplemented|not implemented|not yet implemented|cannot yet|currently unsupported)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GapEntry:
    path: str
    line: int
    text: str


def parse_entries(input_path: Path) -> list[GapEntry]:
    entries: list[GapEntry] = []
    for raw in input_path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.rstrip()
        if not raw:
            continue
        match = ENTRY_RE.match(raw)
        if not match:
            continue
        path, line_str, text = match.groups()
        try:
            line = int(line_str)
        except ValueError:
            continue
        entries.append(GapEntry(path=path, line=line, text=" ".join(text.split())))
    return entries


def classify(entry: GapEntry) -> tuple[str, str, str]:
    """Return (kind, missing, fix)."""
    low_text = entry.text.lower()
    is_test = entry.path.startswith("test/")
    is_lib = entry.path.startswith("lib/")
    is_include = entry.path.startswith("include/")
    is_tooling = (
        entry.path.startswith("tools/")
        or entry.path.startswith("utils/")
        or entry.path.startswith("cmake/")
        or entry.path == "CMakeLists.txt"
    )

    is_todo = bool(TODO_RE.search(entry.text))
    is_unsupported = bool(UNSUPPORTED_RE.search(entry.text))

    test_expectation_markers = (
        "check:",
        "check-not",
        "run:",
        "expected-error",
        "expected-warning",
        "unsupported:",
        "diag:",
        "diag-not",
        "requires:",
    )
    likely_test_expectation = is_test and any(
        marker in low_text for marker in test_expectation_markers
    )

    if likely_test_expectation:
        missing = (
            "This line is primarily a regression expectation for behavior that is "
            "currently unsupported or intentionally skipped, so the underlying "
            "implementation is still incomplete in the product code."
        )
        fix = (
            "Implement or enable the corresponding feature in the owning "
            "production path, then rewrite this test from expected failure/skip "
            "to a positive functional check that validates lowering and runtime "
            "behavior."
        )
        return ("test-expectation", missing, fix)

    if is_todo:
        tail_match = re.search(
            r"\b(?:TODO|FIXME|XXX|TBD|NYI|WIP)\b\s*[:\-]?\s*(.*)",
            entry.text,
            re.IGNORECASE,
        )
        tail = tail_match.group(1).strip() if tail_match else ""
        if tail:
            missing = (
                f"The marker documents unfinished work: {tail}. "
                "The current implementation is partial along this code path."
            )
        else:
            missing = (
                "The marker documents deferred or unfinished work on this path, "
                "which means behavior here is not fully implemented yet."
            )
        fix = (
            "Complete the deferred logic end-to-end in this subsystem, including "
            "type/semantic checks where needed, and add or update regression tests "
            "that fail before the change and pass after it."
        )
        return ("todo", missing, fix)

    if is_unsupported and (is_lib or is_include):
        missing = (
            "The implementation explicitly rejects, drops, or cannot lower this "
            "construct yet, so users still hit an unsupported path."
        )
        fix = (
            "Add full handling for this construct in the owning conversion/dialect "
            "code path, preserve correct diagnostics for invalid cases, and add "
            "positive plus negative tests to lock in behavior."
        )
        return ("implementation-gap", missing, fix)

    if is_unsupported and is_tooling:
        missing = (
            "The tool path hard-restricts this case and reports it as unsupported, "
            "so workflows depending on this mode or op are blocked."
        )
        fix = (
            "Extend the tool's accepted schema/operation set and plumb support "
            "through all downstream call sites, then add CLI/script regression "
            "tests for both supported and invalid inputs."
        )
        return ("tooling-gap", missing, fix)

    if is_unsupported and is_test:
        missing = (
            "This test line references an unsupported behavior, indicating a "
            "known gap in upstream implementation coverage."
        )
        fix = (
            "Implement the missing behavior in production code and update this "
            "test to assert functional semantics rather than unsupported status "
            "once support is complete."
        )
        return ("test-gap-reference", missing, fix)

    # Fallback for weak matches (for example identifiers containing TODO-like text).
    missing = (
        "This marker was captured by the project-wide gap scan and likely "
        "indicates an incomplete or constrained behavior in this location."
    )
    fix = (
        "Confirm intent at this location, implement missing behavior if the marker "
        "is actionable, and add a focused regression test to prevent recurrence."
    )
    return ("unclassified", missing, fix)


def render(entries: list[GapEntry]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    todo_count = sum(1 for e in entries if TODO_RE.search(e.text))
    unsupported_count = sum(1 for e in entries if UNSUPPORTED_RE.search(e.text))

    out: list[str] = []
    out.append("# Project Gap Writeups")
    out.append("")
    out.append(f"Generated: {now}")
    out.append("")
    out.append("This report contains one paragraph per scanned gap marker.")
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append(f"- Total entries: {len(entries)}")
    out.append(f"- TODO/FIXME-like entries: {todo_count}")
    out.append(f"- Unsupported/unimplemented-like entries: {unsupported_count}")
    out.append("")
    out.append("## Gap Paragraphs")
    out.append("")

    for i, entry in enumerate(entries, start=1):
        kind, missing, fix = classify(entry)
        ref = f"{entry.path}:{entry.line}"
        out.append(f"### G{i:04d} `{ref}` ({kind})")
        out.append("")
        out.append(
            f"At `{ref}`, the marker `{entry.text}` indicates a gap signal in this "
            f"path. What's missing: {missing} How to fix: {fix}"
        )
        out.append("")

    return "\n".join(out)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: generate_project_gap_writeups.py <input-rg-list> <output-md>",
            file=sys.stderr,
        )
        return 2

    in_path = Path(argv[1])
    out_path = Path(argv[2])
    entries = parse_entries(in_path)
    out_path.write_text(render(entries), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

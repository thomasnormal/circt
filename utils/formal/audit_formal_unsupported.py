#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Audit unsupported/not-yet-supported diagnostics for formal tooling sources."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def classify(tool: str, rel: str, message: str) -> tuple[str, str]:
    msg = message.lower()
    if tool == "bmc":
        if "multiple clocks" in msg:
            return "multiclock", "WS2"
        if "initial values" in msg and "not yet supported" in msg:
            return "register_init", "WS3"
    if tool == "lec":
        if "aggregate conversion" in msg:
            return "llvm_aggregate_conversion", "WS5"
        if "llhd" in msg and "unsupported" in msg:
            return "llhd_ref_path", "WS4"
    return "other", "WS2-WS5"


def iter_sources(repo_root: Path) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for path in sorted((repo_root / "lib/Tools/circt-bmc").rglob("*.cpp")):
        sources.append(("bmc", path))
    for path in sorted((repo_root / "lib/Tools/circt-bmc").rglob("*.h")):
        sources.append(("bmc", path))
    for path in sorted((repo_root / "lib/Tools/circt-lec").rglob("*.cpp")):
        sources.append(("lec", path))
    for path in sorted((repo_root / "lib/Tools/circt-lec").rglob("*.h")):
        sources.append(("lec", path))
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit formal unsupported/not-yet-supported diagnostics."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current working directory).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output TSV path.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = Path(args.out).resolve()
    pattern = re.compile(r"(unsupported|not yet supported)", re.IGNORECASE)
    rows: list[tuple[str, str, int, str, str, str]] = []
    for tool, path in iter_sources(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines, start=1):
            if not pattern.search(line):
                continue
            message = line.strip()
            category, workstream = classify(tool, rel, message)
            rows.append((tool, rel, idx, category, workstream, message))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "tool\tfile\tline\tcategory\tworkstream\tmessage\n"
        )
        for tool, rel, line_no, category, workstream, message in rows:
            handle.write(
                f"{tool}\t{rel}\t{line_no}\t{category}\t{workstream}\t{message}\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

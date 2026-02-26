#!/usr/bin/env python3
"""Differential illegal-SV checker between Xcelium and CIRCT.

Runs each case in utils/illegal_sv_diff_cases through:
  * Xcelium (`xrun -elaborate` by default)
  * CIRCT (`circt-verilog --no-uvm-auto-include`, mode configurable)

and reports cases that Xcelium rejects while CIRCT accepts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

EXPECT_DIAG_RE = re.compile(r"^\s*//\s*EXPECT_(XCELIUM|CIRCT)_DIAG:\s*(.*?)\s*$")


def sanitize_cell(text: str) -> str:
    return " ".join(text.replace("\t", " ").splitlines()).strip()


def find_repo_root(script_path: Path) -> Path:
    return script_path.resolve().parent.parent


def resolve_tool(explicit: Optional[str], env_var: str, candidates: Iterable[Path], fallback: str) -> str:
    if explicit:
        return explicit
    env_value = os.environ.get(env_var, "")
    if env_value:
        return env_value
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which(fallback)
    if found:
        return found
    return ""


def normalize_tool_invocation(tool: str) -> str:
    # Keep bare tool names for PATH lookup, but canonicalize explicit paths.
    if "/" not in tool:
        return tool
    return str(Path(tool).expanduser().resolve())


def first_diagnostic_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""

    patterns = [
        re.compile(r"\*E,"),
        re.compile(r"\*F,"),
        re.compile(r"\berror\b", re.IGNORECASE),
        re.compile(r"\bfatal\b", re.IGNORECASE),
    ]
    for pattern in patterns:
        for line in lines:
            if pattern.search(line):
                return line
    return lines[0]


def build_xcelium_command(case_path: Path, xcelium_mode: str, xrun: str, xmvlog: str) -> list[str]:
    if xcelium_mode == "xrun-elaborate":
        return [xrun, "-sv", str(case_path), "-elaborate"]
    return [xmvlog, "-sv", str(case_path)]


def build_circt_command(case_path: Path, circt_verilog: str, circt_mode: str) -> list[str]:
    cmd = [circt_verilog, "--no-uvm-auto-include"]
    if circt_mode == "lint-only":
        cmd.append("--lint-only")
    elif circt_mode == "parse-only":
        cmd.append("--parse-only")
    cmd.append(str(case_path))
    return cmd


@dataclass
class CaseExpectations:
    xcelium_diag: list[str]
    circt_diag: list[str]


@dataclass
class ToolRun:
    exit_code: int
    output: str
    timed_out: bool

    @property
    def status(self) -> str:
        if self.timed_out:
            return "timeout"
        return "accept" if self.exit_code == 0 else "reject"


@dataclass
class CaseResult:
    case: str
    expectations: CaseExpectations
    xcelium: ToolRun
    circt: ToolRun
    xcelium_expect_match: Optional[bool]
    circt_expect_match: Optional[bool]

    @property
    def classification(self) -> str:
        xs = self.xcelium.status
        cs = self.circt.status
        if xs == "reject" and cs == "accept":
            return "xcelium_reject_circt_accept"
        if xs == "reject" and cs == "reject":
            return "both_reject"
        if xs == "accept" and cs == "accept":
            return "both_accept"
        if xs == "accept" and cs == "reject":
            return "circt_reject_xcelium_accept"
        return "tool_error"

    @property
    def expectation_status(self) -> str:
        checks = [m for m in (self.xcelium_expect_match, self.circt_expect_match) if m is not None]
        if not checks:
            return "not_checked"
        return "match" if all(checks) else "mismatch"


def read_case_expectations(case_path: Path) -> CaseExpectations:
    xcelium_diag: list[str] = []
    circt_diag: list[str] = []
    for line in case_path.read_text(errors="replace").splitlines():
        match = EXPECT_DIAG_RE.match(line)
        if not match:
            continue
        target, pattern = match.groups()
        pattern = pattern.strip()
        if not pattern:
            continue
        if target == "XCELIUM":
            xcelium_diag.append(pattern)
        else:
            circt_diag.append(pattern)
    return CaseExpectations(xcelium_diag=xcelium_diag, circt_diag=circt_diag)


def check_expected_diagnostics(output: str, expected_patterns: list[str]) -> Optional[bool]:
    if not expected_patterns:
        return None
    output_lower = output.lower()
    for pattern in expected_patterns:
        if pattern.lower() not in output_lower:
            return False
    return True


def run_command(cmd: list[str], cwd: Path, timeout_secs: int) -> ToolRun:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_secs,
            check=False,
        )
        return ToolRun(proc.returncode, proc.stdout, False)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return ToolRun(124, output, True)


def print_case_result(result: CaseResult) -> None:
    expect_suffix = ""
    if result.expectation_status != "not_checked":
        expect_suffix = f" expect={result.expectation_status}"
    print(
        f"[case] {result.case}: xcelium={result.xcelium.status}({result.xcelium.exit_code}) "
        f"circt={result.circt.status}({result.circt.exit_code}) "
        f"class={result.classification}{expect_suffix}"
    )


def write_results_tsv(path: Path, results: list[CaseResult], args: argparse.Namespace) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "case",
                "xcelium_mode",
                "circt_mode",
                "xcelium_exit",
                "circt_exit",
                "xcelium_status",
                "circt_status",
                "classification",
                "expectation_status",
                "expected_xcelium_diag",
                "expected_circt_diag",
                "xcelium_expect_match",
                "circt_expect_match",
                "xcelium_first_diagnostic",
                "circt_first_diagnostic",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.case,
                    args.xcelium_mode,
                    args.circt_mode,
                    result.xcelium.exit_code,
                    result.circt.exit_code,
                    result.xcelium.status,
                    result.circt.status,
                    result.classification,
                    result.expectation_status,
                    " | ".join(result.expectations.xcelium_diag),
                    " | ".join(result.expectations.circt_diag),
                    "" if result.xcelium_expect_match is None else result.xcelium_expect_match,
                    "" if result.circt_expect_match is None else result.circt_expect_match,
                    sanitize_cell(first_diagnostic_line(result.xcelium.output)),
                    sanitize_cell(first_diagnostic_line(result.circt.output)),
                ]
            )


def build_summary_payload(
    *,
    cases_dir: Path,
    args: argparse.Namespace,
    circt_verilog: str,
    xrun: str,
    xmvlog: str,
    summary_counts: dict[str, int],
    expectation_counts: dict[str, int],
    num_cases: int,
) -> dict[str, object]:
    return {
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cases_dir": str(cases_dir),
        "xcelium_mode": args.xcelium_mode,
        "circt_mode": args.circt_mode,
        "circt_verilog": circt_verilog,
        "xrun": xrun,
        "xmvlog": xmvlog,
        "counts": summary_counts,
        "expectation_counts": expectation_counts,
        "num_cases": num_cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=None,
        help="Directory containing illegal-SV case files (*.sv)",
    )
    parser.add_argument(
        "--case-filter",
        default="",
        help="Only run cases whose filename matches this regex",
    )
    parser.add_argument(
        "--xcelium-mode",
        choices=["xrun-elaborate", "xmvlog"],
        default="xrun-elaborate",
        help="Which Xcelium front-end flow to run",
    )
    parser.add_argument("--xrun", default=None, help="Path to xrun")
    parser.add_argument("--xmvlog", default=None, help="Path to xmvlog")
    parser.add_argument("--circt-verilog", default=None, help="Path to circt-verilog")
    parser.add_argument(
        "--circt-mode",
        choices=["full", "lint-only", "parse-only"],
        default="full",
        help="CIRCT import mode to run",
    )
    parser.add_argument(
        "--timeout-secs",
        type=int,
        default=30,
        help="Per-tool timeout per case",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for logs/results (default: /tmp timestamped)",
    )
    parser.add_argument(
        "--fail-on-gap",
        action="store_true",
        help="Return exit 1 if any xcelium_reject_circt_accept cases are found",
    )
    parser.add_argument(
        "--fail-on-xcelium-accept",
        action="store_true",
        help="Return exit 1 if any corpus case is accepted by Xcelium",
    )
    parser.add_argument(
        "--fail-on-expect-mismatch",
        action="store_true",
        help="Return exit 1 if a case's EXPECT_*_DIAG tags do not match tool output",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)
    cases_dir = args.cases_dir or (script_path.parent / "illegal_sv_diff_cases")
    if not cases_dir.is_dir():
        print(f"error: cases directory not found: {cases_dir}", file=sys.stderr)
        return 2

    circt_candidates = [
        repo_root / "build_test/bin/circt-verilog",
        repo_root / "build-slang/bin/circt-verilog",
        repo_root / "build/bin/circt-verilog",
    ]
    circt_verilog = resolve_tool(args.circt_verilog, "CIRCT_VERILOG", circt_candidates, "circt-verilog")
    if not circt_verilog:
        print("error: unable to locate circt-verilog (use --circt-verilog)", file=sys.stderr)
        return 2
    circt_verilog = normalize_tool_invocation(circt_verilog)

    xrun = resolve_tool(args.xrun, "XRUN", [], "xrun")
    xmvlog = resolve_tool(args.xmvlog, "XMVLOG", [], "xmvlog")
    if xrun:
        xrun = normalize_tool_invocation(xrun)
    if xmvlog:
        xmvlog = normalize_tool_invocation(xmvlog)
    if args.xcelium_mode == "xrun-elaborate" and not xrun:
        print("error: unable to locate xrun (use --xrun or set XRUN)", file=sys.stderr)
        return 2
    if args.xcelium_mode == "xmvlog" and not xmvlog:
        print("error: unable to locate xmvlog (use --xmvlog or set XMVLOG)", file=sys.stderr)
        return 2

    case_filter = re.compile(args.case_filter) if args.case_filter else None
    case_paths = sorted(cases_dir.glob("*.sv"))
    if case_filter:
        case_paths = [p for p in case_paths if case_filter.search(p.name)]
    if not case_paths:
        print(f"error: no .sv cases found in {cases_dir}", file=sys.stderr)
        return 2

    out_dir = args.out_dir
    if out_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        out_dir = Path(f"/tmp/circt-illegal-sv-diff-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []

    print(
        f"[illegal-sv-diff] cases={len(case_paths)} xcelium_mode={args.xcelium_mode} "
        f"circt_mode={args.circt_mode} circt={circt_verilog} out_dir={out_dir}"
    )

    for case_path in case_paths:
        case_name = case_path.name
        case_stem = case_path.stem
        case_work_dir = out_dir / "work" / case_stem
        case_work_dir.mkdir(parents=True, exist_ok=True)

        expectations = read_case_expectations(case_path)
        xcelium_cmd = build_xcelium_command(case_path, args.xcelium_mode, xrun, xmvlog)
        circt_cmd = build_circt_command(case_path, circt_verilog, args.circt_mode)

        xcelium_run = run_command(xcelium_cmd, case_work_dir, args.timeout_secs)
        circt_run = run_command(circt_cmd, case_work_dir, args.timeout_secs)

        (case_work_dir / "xcelium.log").write_text(xcelium_run.output)
        (case_work_dir / "circt.log").write_text(circt_run.output)

        case_result = CaseResult(
            case=case_name,
            expectations=expectations,
            xcelium=xcelium_run,
            circt=circt_run,
            xcelium_expect_match=check_expected_diagnostics(
                xcelium_run.output, expectations.xcelium_diag
            ),
            circt_expect_match=check_expected_diagnostics(
                circt_run.output, expectations.circt_diag
            ),
        )
        results.append(case_result)
        print_case_result(case_result)

    summary: dict[str, int] = {}
    expectation_summary: dict[str, int] = {}
    for result in results:
        summary[result.classification] = summary.get(result.classification, 0) + 1
        expectation_summary[result.expectation_status] = (
            expectation_summary.get(result.expectation_status, 0) + 1
        )

    results_tsv = out_dir / "results.tsv"
    write_results_tsv(results_tsv, results, args)

    summary_json = out_dir / "summary.json"
    summary_payload = build_summary_payload(
        cases_dir=cases_dir,
        args=args,
        circt_verilog=circt_verilog,
        xrun=xrun,
        xmvlog=xmvlog,
        summary_counts=summary,
        expectation_counts=expectation_summary,
        num_cases=len(results),
    )
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")

    print("[summary]")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]}")
    print("[expectations]")
    for key in sorted(expectation_summary):
        print(f"  {key}: {expectation_summary[key]}")
    print(f"[artifacts] {results_tsv}")
    print(f"[artifacts] {summary_json}")

    if args.fail_on_xcelium_accept and (
        summary.get("both_accept", 0) > 0 or summary.get("circt_reject_xcelium_accept", 0) > 0
    ):
        return 1
    if args.fail_on_gap and summary.get("xcelium_reject_circt_accept", 0) > 0:
        return 1
    if args.fail_on_expect_mismatch and expectation_summary.get("mismatch", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

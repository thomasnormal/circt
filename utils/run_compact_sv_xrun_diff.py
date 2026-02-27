#!/usr/bin/env python3
"""Generate compact intricate SV stress cases and diff circt-sim vs xrun.

Each generated case:
- Uses case/casez/priority combinational logic and mixed sequential updates.
- Runs for N cycles (default 1000).
- Prints deterministic per-cycle trace lines prefixed with "T ".

The script compiles/runs each case in both simulators and compares trace lines.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class CmdResult:
    exit_code: int
    output: str
    timed_out: bool


@dataclass
class CaseResult:
    case_id: str
    seed: int
    circt_compile: str
    circt_sim: str
    xrun: str
    trace_lines_circt: int
    trace_lines_xrun: int
    match: bool
    mismatch_detail: str
    out_dir: Path


def run_cmd(cmd: Sequence[str], cwd: Path, timeout_s: int) -> CmdResult:
    try:
        p = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        return CmdResult(p.returncode, p.stdout, False)
    except subprocess.TimeoutExpired as e:
        out = ""
        if e.stdout:
            out += e.stdout if isinstance(e.stdout, str) else e.stdout.decode("utf-8", "replace")
        if e.stderr:
            out += e.stderr if isinstance(e.stderr, str) else e.stderr.decode("utf-8", "replace")
        return CmdResult(124, out, True)


def pick(rng: random.Random, options: Sequence[str]) -> str:
    return options[rng.randrange(len(options))]


def generate_case(case_id: str, seed: int, cycles: int) -> str:
    rng = random.Random(seed)

    init0 = rng.getrandbits(32)
    init1 = rng.getrandbits(32)
    init2 = rng.getrandbits(32)
    init3 = rng.getrandbits(32)
    init_sel = rng.getrandbits(8)

    c0 = rng.getrandbits(32)
    c1 = rng.getrandbits(32)
    c2 = rng.getrandbits(32)
    c3 = rng.getrandbits(32)

    rol = rng.randrange(1, 8)
    ror = rng.randrange(1, 8)

    comb_expr_a = [
        "(s0 ^ s1) + {24'h0, sel}",
        "(s0 & s2) ^ (s1 | s3)",
        "{{s0[15:0], s2[31:16]}} + 32'h{:08x}".format(c0),
        "(s3 >> {}) ^ (s2 << {})".format(rng.randrange(1, 6), rng.randrange(1, 6)),
        "~(s0 + s1 + s2 + s3)",
        "((s0 + s2) ^ (s1 - s3)) + 32'h{:08x}".format(c1),
    ]

    comb_expr_b = [
        "s0 + s3",
        "s1 - s2",
        "{s2[7:0], s2[31:8]}",
        "(s0 & s2) | (s1 & s3)",
        "(s0 ^ s2) + (s1 ^ s3)",
        "(s0 << 1) ^ (s3 >> 1)",
    ]

    seq_update_choices = [
        "s0 <= mix(s0, s1, sel); s1 <= mix(s1, s2, sel ^ 8'h3c);",
        "s1 <= mix(s1, s3, sel); s2 <= mix(s2, s0, sel ^ 8'ha5);",
        "s2 <= mix(s2, s3, sel); s3 <= mix(s3, s1, sel ^ 8'h5a);",
        "s3 <= mix(s3, s0, sel); s0 <= mix(s0, s2, sel ^ 8'hc3);",
    ]

    expr_a0 = pick(rng, comb_expr_a)
    expr_a1 = pick(rng, comb_expr_a)
    expr_a2 = pick(rng, comb_expr_a)
    expr_a3 = pick(rng, comb_expr_a)
    expr_a4 = pick(rng, comb_expr_a)

    expr_b0 = pick(rng, comb_expr_b)
    expr_b1 = pick(rng, comb_expr_b)
    expr_b2 = pick(rng, comb_expr_b)
    expr_b3 = pick(rng, comb_expr_b)
    expr_b4 = pick(rng, comb_expr_b)

    seq0 = seq_update_choices[0]
    seq1 = seq_update_choices[1]
    seq2 = seq_update_choices[2]
    seq3 = seq_update_choices[3]

    code = f"""\
`timescale 1ns/1ps

module dut(
  input  logic        clk,
  input  logic        rst_n,
  output logic [31:0] sig
);
  logic [31:0] s0, s1, s2, s3;
  logic [7:0] sel;
  logic [31:0] comb_a, comb_b;

  function automatic [31:0] mix(
    input [31:0] a,
    input [31:0] b,
    input [7:0] c
  );
    reg [31:0] t;
    begin
      t = ((a << (c[2:0])) | (a >> (8 - c[2:0])));
      mix = (t ^ (b + {{24'h0, c}})) + (a & ~b) + 32'h{c2:08x};
    end
  endfunction

  always_comb begin
    unique casez (sel[3:0])
      4'b00??: comb_a = {expr_a0};
      4'b0101: comb_a = {expr_a1};
      4'b011?: comb_a = {expr_a2};
      4'b1?00: comb_a = {expr_a3};
      default: comb_a = {expr_a4};
    endcase

    priority case (sel[2:0])
      3'b000: comb_b = {expr_b0};
      3'b001: comb_b = {expr_b1};
      3'b01?: comb_b = {expr_b2};
      3'b100: comb_b = {expr_b3};
      default: comb_b = {expr_b4};
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s0  <= 32'h{init0:08x};
      s1  <= 32'h{init1:08x};
      s2  <= 32'h{init2:08x};
      s3  <= 32'h{init3:08x};
      sel <= 8'h{init_sel:02x};
      sig <= 32'h0;
    end else begin
      sel <= {{sel[6:0], sel[7] ^ sel[5] ^ sel[4] ^ sel[3]}};
      case (sel[1:0])
        2'b00: begin {seq0} end
        2'b01: begin {seq1} end
        2'b10: begin {seq2} end
        2'b11: begin {seq3} end
      endcase

      sig <= sig
             ^ comb_a
             ^ {{comb_b[15:0], comb_b[31:16]}}
             ^ {{s3[{rol-1}:0], s3[31:{rol}]}}
             ^ (s2 >> {ror})
             ^ 32'h{c3:08x};
    end
  end
endmodule

module top;
  logic clk = 1'b0;
  logic rst_n = 1'b0;
  logic [31:0] sig;
  dut d(.clk(clk), .rst_n(rst_n), .sig(sig));

  always #5 clk = ~clk;

  integer i;
  initial begin
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    for (i = 0; i < {cycles}; i = i + 1) begin
      @(negedge clk);
      if (rst_n)
        $display("T case={case_id} cyc=%0d sig=%08x s0=%08x s1=%08x s2=%08x s3=%08x sel=%02x", i, sig, d.s0, d.s1, d.s2, d.s3, d.sel);
    end
    $display("FINAL case={case_id} sig=%08x s0=%08x s1=%08x s2=%08x s3=%08x sel=%02x", sig, d.s0, d.s1, d.s2, d.s3, d.sel);
    $finish;
  end
endmodule
"""
    return textwrap.dedent(code)


TRACE_RE = re.compile(
    r"T\s+case=(?P<case>\S+)\s+cyc=(?P<cyc>\d+)\s+sig=(?P<sig>[0-9a-fA-F]{8})\s+"
    r"s0=(?P<s0>[0-9a-fA-F]{8})\s+s1=(?P<s1>[0-9a-fA-F]{8})\s+"
    r"s2=(?P<s2>[0-9a-fA-F]{8})\s+s3=(?P<s3>[0-9a-fA-F]{8})\s+sel=(?P<sel>[0-9a-fA-F]{2})"
)

FINAL_RE = re.compile(
    r"FINAL\s+case=(?P<case>\S+)\s+sig=(?P<sig>[0-9a-fA-F]{8})\s+"
    r"s0=(?P<s0>[0-9a-fA-F]{8})\s+s1=(?P<s1>[0-9a-fA-F]{8})\s+"
    r"s2=(?P<s2>[0-9a-fA-F]{8})\s+s3=(?P<s3>[0-9a-fA-F]{8})\s+sel=(?P<sel>[0-9a-fA-F]{2})"
)


def _normalize_output_for_parse(output: str) -> str:
    # Remove simulator diagnostics that may appear as standalone lines or be
    # interleaved into $display output mid-line.
    text = re.sub(r"\[circt-sim\][^\n]*\n?", "", output)
    cleaned_lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(cleaned_lines)


def extract_trace_lines(output: str) -> list[str]:
    text = _normalize_output_for_parse(output)
    out = []
    for m in TRACE_RE.finditer(text):
        out.append(
            "T case={case} cyc={cyc} sig={sig} s0={s0} s1={s1} s2={s2} s3={s3} sel={sel}".format(
                **m.groupdict()
            )
        )
    return out


def extract_final_line(output: str) -> str:
    text = _normalize_output_for_parse(output)
    finals = list(FINAL_RE.finditer(text))
    if not finals:
        return ""
    gd = finals[-1].groupdict()
    return "FINAL case={case} sig={sig} s0={s0} s1={s1} s2={s2} s3={s3} sel={sel}".format(
        **gd
    )


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_case(
    case_id: str,
    seed: int,
    cycles: int,
    out_root: Path,
    circt_verilog: str,
    circt_sim: str,
    xrun: str,
    compile_timeout: int,
    sim_timeout: int,
) -> CaseResult:
    case_dir = out_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    sv = case_dir / f"{case_id}.sv"
    mlir = case_dir / f"{case_id}.mlir"
    circt_compile_log = case_dir / "circt-verilog.log"
    circt_sim_log = case_dir / "circt-sim.log"
    xrun_log = case_dir / "xrun.log"

    write_file(sv, generate_case(case_id, seed, cycles))

    cver_cmd = [
        circt_verilog,
        str(sv),
        "--ir-llhd",
        "--no-uvm-auto-include",
        "--top=top",
        "-o",
        str(mlir),
    ]
    cver = run_cmd(cver_cmd, case_dir, compile_timeout)
    circt_compile_log.write_text(cver.output, encoding="utf-8")
    if cver.exit_code != 0:
        return CaseResult(
            case_id=case_id,
            seed=seed,
            circt_compile=f"FAIL({cver.exit_code})",
            circt_sim="SKIP",
            xrun="SKIP",
            trace_lines_circt=0,
            trace_lines_xrun=0,
            match=False,
            mismatch_detail="circt-verilog failed",
            out_dir=case_dir,
        )

    if not mlir.exists():
        return CaseResult(
            case_id=case_id,
            seed=seed,
            circt_compile="FAIL(no-mlir)",
            circt_sim="SKIP",
            xrun="SKIP",
            trace_lines_circt=0,
            trace_lines_xrun=0,
            match=False,
            mismatch_detail="circt-verilog produced no MLIR output file",
            out_dir=case_dir,
        )

    csim_cmd = [
        circt_sim,
        str(mlir),
        "--top=top",
        "--max-time=20000000000",
    ]
    csim = run_cmd(csim_cmd, case_dir, sim_timeout)
    circt_sim_log.write_text(csim.output, encoding="utf-8")

    if csim.exit_code != 0:
        return CaseResult(
            case_id=case_id,
            seed=seed,
            circt_compile="OK",
            circt_sim=f"FAIL({csim.exit_code})",
            xrun="SKIP",
            trace_lines_circt=0,
            trace_lines_xrun=0,
            match=False,
            mismatch_detail="circt-sim failed",
            out_dir=case_dir,
        )

    # Run xrun with a short tcl script to keep behavior deterministic.
    run_ns = max(20000, (cycles + 20) * 10)
    tcl = case_dir / "run.tcl"
    write_file(
        tcl,
        "\n".join(
            [
                "run {}ns".format(run_ns),
                "exit",
                "",
            ]
        ),
    )
    xrun_cmd = [
        xrun,
        "-64bit",
        "-sv",
        "-access",
        "+rwc",
        "-input",
        str(tcl.name),
        str(sv.name),
        "-top",
        "top",
    ]
    xsim = run_cmd(xrun_cmd, case_dir, sim_timeout)
    xrun_log.write_text(xsim.output, encoding="utf-8")

    if xsim.exit_code != 0:
        return CaseResult(
            case_id=case_id,
            seed=seed,
            circt_compile="OK",
            circt_sim="OK",
            xrun=f"FAIL({xsim.exit_code})",
            trace_lines_circt=0,
            trace_lines_xrun=0,
            match=False,
            mismatch_detail="xrun failed",
            out_dir=case_dir,
        )

    circt_trace = extract_trace_lines(csim.output)
    xrun_trace = extract_trace_lines(xsim.output)
    circt_final = extract_final_line(csim.output)
    xrun_final = extract_final_line(xsim.output)

    mismatch_detail = ""
    match = True

    if len(circt_trace) != cycles:
        match = False
        mismatch_detail = f"circt trace length {len(circt_trace)} != expected {cycles}"
    elif len(xrun_trace) != cycles:
        match = False
        mismatch_detail = f"xrun trace length {len(xrun_trace)} != expected {cycles}"
    else:
        for i, (lhs, rhs) in enumerate(zip(circt_trace, xrun_trace)):
            if lhs != rhs:
                match = False
                mismatch_detail = f"trace mismatch at cycle {i}: circt='{lhs}' xrun='{rhs}'"
                break

    if match and circt_final != xrun_final:
        match = False
        mismatch_detail = f"final mismatch: circt='{circt_final}' xrun='{xrun_final}'"

    if match:
        mismatch_detail = "OK"

    return CaseResult(
        case_id=case_id,
        seed=seed,
        circt_compile="OK",
        circt_sim="OK",
        xrun="OK",
        trace_lines_circt=len(circt_trace),
        trace_lines_xrun=len(xrun_trace),
        match=match,
        mismatch_detail=mismatch_detail,
        out_dir=case_dir,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=12)
    p.add_argument("--cycles", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260226)
    p.add_argument("--out-dir", default="/tmp/circt-compact-sv-diff")
    p.add_argument("--circt-verilog", default="build_test/bin/circt-verilog")
    p.add_argument("--circt-sim", default="build_test/bin/circt-sim")
    p.add_argument("--xrun", default="xrun")
    p.add_argument("--compile-timeout", type=int, default=120)
    p.add_argument("--sim-timeout", type=int, default=180)
    p.add_argument("--stop-on-first-mismatch", action="store_true")
    return p.parse_args()


def resolve_tool(spec: str) -> str | None:
    if os.sep in spec:
        p = Path(spec).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return str(p) if p.exists() else None
    return shutil.which(spec)


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    circt_verilog = resolve_tool(args.circt_verilog)
    circt_sim = resolve_tool(args.circt_sim)
    xrun = resolve_tool(args.xrun)

    if not circt_verilog:
        print(f"error: circt-verilog not found: {args.circt_verilog}", file=sys.stderr)
        return 2
    if not circt_sim:
        print(f"error: circt-sim not found: {args.circt_sim}", file=sys.stderr)
        return 2
    if not xrun:
        print(f"error: xrun not found: {args.xrun}", file=sys.stderr)
        return 2

    results: list[CaseResult] = []
    mismatches = 0

    print(f"[compact-sv-diff] out_dir={out_dir}")
    print(
        f"[compact-sv-diff] cases={args.cases} cycles={args.cycles} seed={args.seed} "
        f"circt_verilog={circt_verilog} circt_sim={circt_sim} xrun={xrun}"
    )

    for i in range(args.cases):
        case_seed = args.seed + i * 9973
        case_id = f"case_{i:03d}"
        print(f"[compact-sv-diff] running {case_id} seed={case_seed}")
        res = run_case(
            case_id=case_id,
            seed=case_seed,
            cycles=args.cycles,
            out_root=out_dir,
            circt_verilog=circt_verilog,
            circt_sim=circt_sim,
            xrun=xrun,
            compile_timeout=args.compile_timeout,
            sim_timeout=args.sim_timeout,
        )
        results.append(res)
        if res.match:
            print(f"[compact-sv-diff] {case_id} OK")
        else:
            mismatches += 1
            print(f"[compact-sv-diff] {case_id} MISMATCH: {res.mismatch_detail}")
            if args.stop_on_first_mismatch:
                break

    tsv = out_dir / "summary.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "case",
                "seed",
                "circt_compile",
                "circt_sim",
                "xrun",
                "trace_lines_circt",
                "trace_lines_xrun",
                "match",
                "detail",
                "dir",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.case_id,
                    r.seed,
                    r.circt_compile,
                    r.circt_sim,
                    r.xrun,
                    r.trace_lines_circt,
                    r.trace_lines_xrun,
                    "1" if r.match else "0",
                    r.mismatch_detail,
                    str(r.out_dir),
                ]
            )

    print(f"[compact-sv-diff] summary_tsv={tsv}")
    print(f"[compact-sv-diff] total={len(results)} mismatches={mismatches}")

    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

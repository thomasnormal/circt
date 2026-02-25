#!/usr/bin/env python3
"""Stable local wrapper for CVDP cocotb runs.

Loads the benchmark runner from ~/cvdp_benchmark, applies a safer cocotb
compatibility patcher, and uses strict timeout classification to avoid turning
functional cocotb mismatches into infrastructure failures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Tuple


def load_base_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cvdp_base_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load runner module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_cocotb_compat_source(text: str) -> str:
    """Apply cocotb v1->v2 compatibility rewrites safely."""
    orig = text

    if "from cocotb.binary import" in text:
        text = text.replace(
            "from cocotb.binary import BinaryValue",
            "try:\n    from cocotb.binary import BinaryValue\n"
            "except ImportError:\n    from cocotb.types import LogicArray as BinaryValue",
        )

    text = text.replace(
        "from cocotb.result import TestFailure",
        "try:\n    from cocotb.result import TestFailure\n"
        "except ImportError:\n    TestFailure = AssertionError",
    )
    text = text.replace(
        "from cocotb.result import TestSuccess",
        "try:\n    from cocotb.result import TestSuccess\n"
        "except ImportError:\n    class TestSuccess(Exception): pass",
    )

    text = re.sub(
        r"@cocotb\.coroutine\s*\n(\s*)(async\s+def|def)\s+",
        r"\1async def ",
        text,
    )

    text = text.replace(
        "import cocotb.result",
        "try:\n    import cocotb.result\nexcept ImportError:\n    pass",
    )

    has_tools_runner = bool(
        re.search(
            r"(?m)^\s*from\s+cocotb_tools\.runner\s+import\s+get_runner\s*$",
            text,
        )
    )
    if not has_tools_runner:
        text = re.sub(
            r"(?m)^([ \t]*)from\s+cocotb\.runner\s+import\s+get_runner\s*$",
            r"\1try:\n\1    from cocotb_tools.runner import get_runner\n"
            r"\1except ImportError:\n\1    from cocotb.runner import get_runner",
            text,
        )

    if ".to_unsigned()" in text or ".integer" in text or ".signed_integer" in text:
        text = re.sub(
            r"(\w+(?:\.\w+)*)\.value\.to_unsigned\(\)",
            r"int(\1.value)",
            text,
        )
        text = re.sub(
            r"(\w+(?:\.\w+)*)\.value\.integer",
            r"int(\1.value)",
            text,
        )
        text = re.sub(
            r"(\w+(?:\.\w+)*)\.value\.signed_integer",
            r"int(\1.value)",
            text,
        )

    packed_shim = (
        "\n# cocotb v2: restore packed vector bit-indexing (read-only)\n"
        "try:\n"
        "    from cocotb.handle import LogicArrayObject as _LAO\n"
        "    if not hasattr(_LAO, '_v1_getitem_patched'):\n"
        "        _LAO.__getitem__ = lambda self, idx: self.value[idx]\n"
        "        _LAO._v1_getitem_patched = True\n"
        "except (ImportError, AttributeError):\n"
        "    pass\n"
    )
    if "import cocotb" in text and packed_shim not in text:
        text = packed_shim + text

    return text if text != orig else orig


def patch_cocotb_compat_file(pyfile: Path) -> None:
    text = pyfile.read_text()
    patched = patch_cocotb_compat_source(text)
    if patched != text:
        pyfile.write_text(patched)


def classify_result(ok: bool, passes: int, fails: int, output: str) -> str:
    """Classify the simulation outcome."""
    if output.startswith("TIMEOUT\n") or output.strip() == "TIMEOUT":
        if "[circt-sim] Stage: init" in output or "cocotb                             Running tests" in output:
            return "COCOTB_FAIL"
        return "SIM_TIMEOUT"
    if "resource guard triggered" in output:
        return "COCOTB_FAIL"
    # Harness/test errors should count as functional mismatch, not infra failure.
    if "IndentationError:" in output or "SyntaxError:" in output:
        return "COCOTB_FAIL"
    if "Traceback (most recent call last):" in output and (
        "cocotb" in output or "pygpi" in output or "gpi" in output
    ):
        return "COCOTB_FAIL"
    if ok:
        return "COCOTB_PASS"
    if passes > 0 or fails > 0:
        return "COCOTB_FAIL"
    return "SIM_FAIL"


def write_stub_sv(problem_dir: Path, top_module: str) -> Path:
    """Materialize a minimal stub SV design for harness-only infra checks.

    CVDP v1.0.2 public datasets often redact the referenced RTL sources but keep
    the cocotb harness. For infra-level coverage, we compile/run a stub module
    with common clock/reset names so cocotb can at least connect to a DUT.
    """
    stub = problem_dir / "_cvdp_stub.sv"
    stub.write_text(
        "\n".join(
            [
                "`timescale 1ns/1ps",
                "",
                f"module {top_module};",
                "  logic clk = 0;",
                "  logic rst = 1;",
                "  logic reset = 1;",
                "  logic rst_n = 0;",
                "  logic valid = 0;",
                "  logic ready = 1;",
                "  logic [31:0] data = 32'h0;",
                "",
                "  // Provide a free-running clock and basic reset sequencing.",
                "  always #5 clk = ~clk;",
                "  initial begin",
                "    #20;",
                "    rst = 0;",
                "    reset = 0;",
                "    rst_n = 1;",
                "    #20;",
                "    valid = 1;",
                "    data = 32'h1234_5678;",
                "  end",
                "endmodule",
                "",
            ]
        )
    )
    return stub


def infer_top_and_test_module(
    env_vars: dict, py_files: list[Path], sv_files: list[Path]
) -> Tuple[str, str]:
    top_module = env_vars.get("TOPLEVEL", "")
    test_module = env_vars.get("MODULE", "")

    if not top_module or not test_module:
        for pyf in py_files:
            if pyf.name.startswith("test_") and pyf.name != "test_runner.py":
                test_module = test_module or pyf.stem
        if not top_module:
            for svf in sv_files:
                modules = re.findall(r"^\s*module\s+(\w+)", svf.read_text(), re.MULTILINE)
                if modules:
                    top_module = top_module or modules[0]
    return top_module, test_module


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CVDP cocotb tests with circt-sim")
    parser.add_argument("-f", "--file", required=True, help="JSONL dataset file")
    parser.add_argument("-n", "--limit", type=int, default=0, help="Max problems (0=all)")
    parser.add_argument("-i", "--id", help="Test only this problem ID")
    parser.add_argument("-o", "--output", default="", help="Output directory")
    parser.add_argument("--compile-only", action="store_true", help="Only compile, don't simulate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    base_runner = Path(
        os.environ.get(
            "CVDP_COCOTB_BASE_RUNNER", "~/cvdp_benchmark/run_circt_cocotb.py"
        )
    ).expanduser()
    if not base_runner.exists():
        raise FileNotFoundError(f"base runner not found: {base_runner}")
    base = load_base_runner(base_runner)
    base._patch_cocotb_compat = patch_cocotb_compat_file

    compile_timeout = os.environ.get("CVDP_COMPILE_TIMEOUT")
    if compile_timeout:
        base.COMPILE_TIMEOUT = int(compile_timeout)
    sim_timeout = os.environ.get("CVDP_SIM_TIMEOUT")
    if sim_timeout:
        base.SIM_TIMEOUT = int(sim_timeout)

    work_dir = Path(args.output) if args.output else Path(base.WORK_DIR)
    work_dir.mkdir(parents=True, exist_ok=True)

    problems = []
    with open(args.file) as f:
        for line in f:
            d = json.loads(line)
            if args.id and d["id"] != args.id:
                continue
            problems.append(d)
    if args.limit > 0:
        problems = problems[: args.limit]

    print(f"Testing {len(problems)} CVDP problems with circt-sim + cocotb...")
    results = {
        "compile_pass": [],
        "compile_fail": [],
        "no_sv": [],
        "stub_sv": [],
        "sim_pass": [],
        "sim_fail": [],
        "sim_timeout": [],
        "cocotb_pass": [],
        "cocotb_fail": [],
    }
    start = time.time()

    for i, p in enumerate(problems):
        pid = p["id"]
        m = re.match(r"cvdp_copilot_(.+)_(\d+)", pid)
        if not m:
            print(f"  [{i + 1}/{len(problems)}] [SKIP] {pid}")
            continue

        name, num = m.groups()
        problem_dir = work_dir / name / num
        problem_dir.mkdir(parents=True, exist_ok=True)

        sv_files, py_files, env_vars = base.extract_files(p, problem_dir)
        stubbed = False
        if not sv_files:
            # Some CVDP datapoints keep the cocotb harness + .env but redact the
            # referenced SV. For infra-level coverage, synthesize a stub SV
            # module with the requested toplevel name.
            top_module, test_module = infer_top_and_test_module(env_vars, py_files, sv_files)
            if top_module and test_module:
                stub_path = write_stub_sv(problem_dir, top_module=top_module)
                sv_files = [stub_path]
                stubbed = True
                results["no_sv"].append(pid)
                results["stub_sv"].append(pid)
                print(f"  [{i + 1}/{len(problems)}] [STUB_SV] {pid}", flush=True)
            else:
                results["no_sv"].append(pid)
                print(f"  [{i + 1}/{len(problems)}] [NO_SV] {pid}", flush=True)
                continue

        output_mlir = problem_dir / "output.mlir"
        if output_mlir.exists() and output_mlir.stat().st_size > 100:
            ok, stderr = True, "(reused existing MLIR)"
        else:
            ok, stderr = base.compile_sv(sv_files, output_mlir)
            (problem_dir / "compile.log").write_text(stderr or "")

        if not ok:
            results["compile_fail"].append(pid)
            print(f"  [{i + 1}/{len(problems)}] [COMPILE_FAIL] {pid}", flush=True)
            if args.verbose:
                for line in (stderr or "").split("\n"):
                    if "error:" in line.lower():
                        print(f"    {line.strip()[:120]}")
                        break
            continue
        results["compile_pass"].append(pid)

        if args.compile_only:
            print(f"  [{i + 1}/{len(problems)}] [COMPILE_OK] {pid}", flush=True)
            continue

        top_module, test_module = infer_top_and_test_module(env_vars, py_files, sv_files)
        if not top_module or not test_module:
            print(
                f"  [{i + 1}/{len(problems)}] [COMPILE_OK] {pid} "
                "(can't determine top/test module)"
            )
            continue

        if output_mlir.exists():
            mlir_text = output_mlir.read_text()
            available_modules = re.findall(r"hw\.module\s+@(\w+)", mlir_text)
            if top_module not in available_modules and available_modules:
                orig_top = top_module
                top_module = available_modules[0]
                if args.verbose:
                    print(f"    (top '{orig_top}' not found, using '{top_module}')")

        test_dir = problem_dir / "src"
        if not test_dir.exists():
            test_dir = problem_dir

        # Keep stubbed harnesses bounded; these runs are infra-oriented.
        max_time_fs = None
        if stubbed:
            try:
                max_time_fs = int(os.environ.get("CVDP_STUB_MAX_TIME_FS", "2000000000"))
            except ValueError:
                max_time_fs = 2000000000
        try:
            if max_time_fs is None:
                ok, passes, fails, output = base.run_cocotb_sim(
                    output_mlir, top_module, test_module, test_dir, sv_files=sv_files
                )
            else:
                ok, passes, fails, output = base.run_cocotb_sim(
                    output_mlir,
                    top_module,
                    test_module,
                    test_dir,
                    sv_files=sv_files,
                    max_time_fs=max_time_fs,
                )
        except TypeError:
            # Older base runners may not accept max_time_fs; best-effort.
            ok, passes, fails, output = base.run_cocotb_sim(
                output_mlir, top_module, test_module, test_dir, sv_files=sv_files
            )
        (problem_dir / "sim.log").write_text(output)

        status = classify_result(ok=ok, passes=passes, fails=fails, output=output)
        if status == "SIM_TIMEOUT":
            results["sim_timeout"].append(pid)
            label = "SIM_TIMEOUT"
        elif status == "COCOTB_PASS":
            results["cocotb_pass"].append(pid)
            label = f"COCOTB_PASS ({passes} tests)"
        elif status == "COCOTB_FAIL":
            results["cocotb_fail"].append(pid)
            label = f"COCOTB_FAIL ({passes}P/{fails}F)"
        else:
            results["sim_fail"].append(pid)
            label = "SIM_FAIL"

        print(f"  [{i + 1}/{len(problems)}] [{label}] {pid}", flush=True)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  COMPILE_OK:     {len(results['compile_pass'])}/{len(problems)}")
    print(f"  COMPILE_FAIL:   {len(results['compile_fail'])}/{len(problems)}")
    print(f"  NO_SV:          {len(results['no_sv'])}/{len(problems)}")
    print(f"  STUB_SV:        {len(results['stub_sv'])}/{len(problems)}")
    if not args.compile_only:
        print(f"  COCOTB_PASS:    {len(results['cocotb_pass'])}")
        print(f"  COCOTB_FAIL:    {len(results['cocotb_fail'])}")
        print(f"  SIM_FAIL:       {len(results['sim_fail'])}")
        print(f"  SIM_TIMEOUT:    {len(results['sim_timeout'])}")

    results_file = work_dir / "circt_cocotb_results.json"
    with open(results_file, "w") as f:
        json.dump({"results": results, "elapsed": elapsed}, f, indent=2)
    print(f"\nResults saved to {results_file}")

    if results["cocotb_fail"] or results["sim_fail"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

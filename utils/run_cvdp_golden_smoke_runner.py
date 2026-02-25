#!/usr/bin/env python3
"""Stable wrapper for CVDP golden smoke runner.

Snapshots circt-verilog/circt-sim before invoking the benchmark script so lane
execution is not affected by concurrent tool rebuilds replacing binaries.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import sys
from pathlib import Path
from types import ModuleType


def load_base_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cvdp_golden_base_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load runner module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def snapshot_tool(src: Path, snapshot_dir: Path, dest_name: str) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"tool not found: {src}")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    dest = snapshot_dir / dest_name
    shutil.copy2(src, dest)
    make_executable(dest)
    return dest


def resolve_tool_path(src: Path) -> Path:
    if src.exists():
        return src
    # Historical scripts used "build-test" while local trees use "build_test".
    # Accept both spellings to keep unified smoke lanes resilient.
    src_text = str(src)
    if "build-test" in src_text:
        alias = Path(src_text.replace("build-test", "build_test"))
        if alias.exists():
            return alias
    raise FileNotFoundError(f"tool not found: {src}")


def resolve_output_dir(args: list[str]) -> Path:
    out = ""
    for i, arg in enumerate(args):
        if arg in ("-o", "--output") and i + 1 < len(args):
            out = args[i + 1]
            break
    if not out:
        out = os.environ.get("CVDP_GOLDEN_OUT_DIR", "circt_work_golden")
    return Path(out).expanduser()


def run_base(base: ModuleType, base_path: Path, passthrough_args: list[str]) -> int:
    old_argv = sys.argv
    sys.argv = [str(base_path)] + passthrough_args
    try:
        base.main()
        return 0
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    finally:
        sys.argv = old_argv


def main() -> int:
    passthrough_args = sys.argv[1:]
    out_dir = resolve_output_dir(passthrough_args)
    snapshot_dir = out_dir / ".tool-snapshot"

    base_path = Path(
        os.environ.get(
            "CVDP_GOLDEN_BASE_RUNNER", "~/cvdp_benchmark/run_circt_golden_smoke.py"
        )
    ).expanduser()
    if not base_path.exists():
        raise FileNotFoundError(f"base runner not found: {base_path}")

    base = load_base_runner(base_path)
    verilog = Path(str(getattr(base, "CIRCT_VERILOG"))).expanduser()
    sim = Path(str(getattr(base, "CIRCT_SIM"))).expanduser()

    if os.environ.get("CVDP_GOLDEN_CIRCT_VERILOG"):
        verilog = Path(os.environ["CVDP_GOLDEN_CIRCT_VERILOG"]).expanduser()
    if os.environ.get("CVDP_GOLDEN_CIRCT_SIM"):
        sim = Path(os.environ["CVDP_GOLDEN_CIRCT_SIM"]).expanduser()

    verilog = resolve_tool_path(verilog)
    sim = resolve_tool_path(sim)
    snapshot_verilog = snapshot_tool(verilog, snapshot_dir, "circt-verilog")
    snapshot_sim = snapshot_tool(sim, snapshot_dir, "circt-sim")
    base.CIRCT_VERILOG = snapshot_verilog
    base.CIRCT_SIM = snapshot_sim

    if os.environ.get("CVDP_GOLDEN_COMPILE_TIMEOUT"):
        base.COMPILE_TIMEOUT = int(os.environ["CVDP_GOLDEN_COMPILE_TIMEOUT"])
    if os.environ.get("CVDP_GOLDEN_SIM_TIMEOUT"):
        base.SIM_TIMEOUT = int(os.environ["CVDP_GOLDEN_SIM_TIMEOUT"])

    return run_base(base, base_path, passthrough_args)


if __name__ == "__main__":
    raise SystemExit(main())

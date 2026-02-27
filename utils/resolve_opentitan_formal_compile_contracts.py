#!/usr/bin/env python3
"""Resolve OpenTitan FPV compile contracts from a selected-target manifest.

This script consumes the TSV manifest produced by
`utils/select_opentitan_formal_cfgs.py` and, per target, runs a FuseSoC setup
step to obtain an EDA description (`*.eda.yml`). It then emits deterministic
compile-contract rows with normalized file/define/include fingerprints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ManifestRow:
    target_name: str
    fusesoc_core: str
    task: str
    stopats: tuple[str, ...]
    flow: str
    sub_flow: str
    rel_path: str


@dataclass(frozen=True)
class TaskPolicy:
    task_profile: str
    task_known: str
    stopat_mode: str
    blackbox_policy: str
    task_policy_fingerprint: str


KNOWN_TASK_POLICIES: dict[str, tuple[str, str, str]] = {
    "": ("fpv_default", "none", "none"),
    "FpvDefault": ("fpv_default", "none", "none"),
    "FpvSecCm": ("fpv_sec_cm", "task_defined", "prim_count,prim_double_lfsr"),
    "SecCmCFI": ("sec_cm_cfi", "task_defined", "none"),
    "SecCmCFILinear": ("sec_cm_cfi_linear", "task_defined", "none"),
    "PwrmgrSecCmEsc": ("pwrmgr_sec_cm_esc", "none", "none"),
}

TARGET_DEFINE_OVERRIDES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"^aes_masked_"), ("EN_MASKING=1",)),
    (re.compile(r"^aes_unmasked_"), ("EN_MASKING=0",)),
)


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Input target manifest TSV")
    parser.add_argument(
        "--opentitan-root",
        required=True,
        help="OpenTitan checkout root (used as --cores-root for FuseSoC)",
    )
    parser.add_argument(
        "--out-contracts",
        required=True,
        help="Output compile-contract TSV path",
    )
    parser.add_argument(
        "--workdir",
        default="",
        help="Optional persistent workdir root (default: temp dir)",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep auto-generated temporary workdirs",
    )
    parser.add_argument(
        "--fusesoc-bin",
        default="fusesoc",
        help="FuseSoC executable (default: fusesoc in PATH)",
    )
    parser.add_argument(
        "--fusesoc-target",
        default="formal",
        help="FuseSoC target name (default: formal)",
    )
    parser.add_argument(
        "--fusesoc-tool",
        default="symbiyosys",
        help="FuseSoC tool name (default: symbiyosys)",
    )
    parser.add_argument(
        "--target-filter",
        default="",
        help=(
            "Optional regex filter on manifest target_name rows before "
            "FuseSoC setup/contract resolution"
        ),
    )
    parser.add_argument(
        "--fail-on-unknown-task",
        action="store_true",
        help=(
            "Fail when manifest rows contain OpenTitan FPV task names that are "
            "not recognized by the CIRCT task-policy adapter"
        ),
    )
    return parser.parse_args()


def sanitize_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", token)


def read_manifest(path: Path) -> list[ManifestRow]:
    if not path.is_file():
        fail(f"manifest not found: {path}")
    rows: list[ManifestRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"manifest is missing header row: {path}")
        required = {"target_name", "fusesoc_core"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            fail(
                f"manifest missing required columns {missing}: {path} "
                f"(found: {reader.fieldnames})"
            )
        for idx, row in enumerate(reader, start=2):
            target_name = (row.get("target_name") or "").strip()
            fusesoc_core = (row.get("fusesoc_core") or "").strip()
            if not target_name and not fusesoc_core:
                continue
            if not target_name or not fusesoc_core:
                fail(
                    f"manifest row {idx} missing target_name/fusesoc_core in {path}"
                )
            rows.append(
                ManifestRow(
                    target_name=target_name,
                    fusesoc_core=fusesoc_core,
                    task=(row.get("task") or "").strip(),
                    stopats=parse_manifest_stopats(row.get("stopats") or ""),
                    flow=(row.get("flow") or "").strip(),
                    sub_flow=(row.get("sub_flow") or "").strip(),
                    rel_path=(row.get("rel_path") or "").strip(),
                )
            )
    if not rows:
        fail(f"manifest has no target rows: {path}")
    return rows


def locate_eda_yml(job_dir: Path) -> Path | None:
    candidates = sorted(
        job_dir.glob("build/**/*.eda.yml"),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0]


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if isinstance(value, list):
        out: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                token = entry.strip()
                if token:
                    out.append(token)
            else:
                out.append(str(entry))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for key in sorted(value.keys()):
            raw = value[key]
            if raw is None or raw == "":
                out.append(str(key))
            else:
                out.append(f"{key}={raw}")
        return out
    return [str(value)]


def normalize_stopats(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        token = value.strip()
        return (token,) if token else ()
    if isinstance(value, list):
        out: list[str] = []
        for entry in value:
            token = str(entry).strip()
            if token:
                out.append(token)
        return tuple(out)
    token = str(value).strip()
    return (token,) if token else ()


def parse_manifest_stopats(raw: str) -> tuple[str, ...]:
    token = raw.strip()
    if not token:
        return ()
    try:
        parsed = json.loads(token)
    except json.JSONDecodeError:
        return normalize_stopats(token)
    return normalize_stopats(parsed)


def toplevel_string(value: Any) -> str:
    parts = normalize_string_list(value)
    return ",".join(parts)


VERILOG_SOURCE_SUFFIXES = {".sv", ".v"}
VERILOG_HEADER_SUFFIXES = {".svh", ".vh"}
MODULE_DECL_RE = re.compile(
    r"(?m)^\s*module(?:\s+automatic)?\s+([A-Za-z_][A-Za-z0-9_$]*)\b"
)


def classify_verilog_entry(file_type: Any, path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VERILOG_SOURCE_SUFFIXES:
        return "source"
    if suffix in VERILOG_HEADER_SUFFIXES:
        return "header"
    token = str(file_type or "").strip().lower()
    if "verilog" not in token:
        return ""
    return "source"


def resolve_eda_paths(entries: Any, eda_dir: Path) -> tuple[list[str], list[str]]:
    if not isinstance(entries, list):
        return [], []
    files: list[str] = []
    include_dirs: list[str] = []
    seen_incdirs: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        file_path = (eda_dir / name).resolve(strict=False)
        file_text = str(file_path)
        verilog_kind = classify_verilog_entry(entry.get("file_type"), file_path)
        if not verilog_kind:
            continue
        if entry.get("is_include_file"):
            incdir = str(file_path.parent)
            if incdir not in seen_incdirs:
                include_dirs.append(incdir)
                seen_incdirs.add(incdir)
            # Keep include-marked Verilog source files in the compile unit.
            # OpenTitan/formal flows rely on some `.sv` macro libraries
            # (for example `prim_assert.sv`) being compiled, while header-only
            # include files (`.svh`, `.vh`) should remain include-only.
            if verilog_kind != "source":
                continue
        elif verilog_kind == "header":
            incdir = str(file_path.parent)
            if incdir not in seen_incdirs:
                include_dirs.append(incdir)
                seen_incdirs.add(incdir)
            continue
        files.append(file_text)
    return files, include_dirs


def apply_platform_file_filter(rel_path: str, files: list[str]) -> list[str]:
    rel = rel_path.lower()
    if "top_earlgrey" in rel:
        return [item for item in files if "/lowrisc_englishbreakfast_" not in item]
    if "top_englishbreakfast" in rel:
        return [item for item in files if "/lowrisc_earlgrey_" not in item]
    return files


def collect_declared_module_names(files: list[str]) -> set[str]:
    names: set[str] = set()
    for item in files:
        path = Path(item)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in MODULE_DECL_RE.finditer(text):
            names.add(match.group(1))
    return names


def resolve_contract_toplevel(raw_toplevel: str, files: list[str]) -> str:
    toplevel = raw_toplevel.strip()
    if not toplevel or "," in toplevel:
        return raw_toplevel
    module_names = collect_declared_module_names(files)
    if toplevel in module_names:
        return toplevel
    if toplevel.endswith("_tb"):
        fallback = toplevel[:-3]
        if fallback and fallback in module_names:
            return fallback
    return raw_toplevel


def define_key(token: str) -> str:
    text = token.strip()
    if not text:
        return ""
    return text.split("=", 1)[0].strip()


def apply_target_define_overrides(target_name: str, defines: list[str]) -> list[str]:
    merged = list(defines)
    keys = {define_key(item) for item in merged if define_key(item)}
    for pattern, extras in TARGET_DEFINE_OVERRIDES:
        if not pattern.search(target_name):
            continue
        for extra in extras:
            key = define_key(extra)
            if not key or key in keys:
                continue
            merged.append(extra)
            keys.add(key)
    return merged


def hash_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_task_policy(task: str, stopats: tuple[str, ...]) -> TaskPolicy:
    raw_task = task.strip()
    policy = KNOWN_TASK_POLICIES.get(raw_task)
    if policy is None:
        task_profile = "unknown_task"
        task_known = "0"
        stopat_mode = "task_defined"
        blackbox_policy = "none"
    else:
        task_profile, stopat_mode, blackbox_policy = policy
        task_known = "1"
    digest = hashlib.sha256()
    digest.update(f"task={raw_task}\n".encode("utf-8"))
    digest.update(f"profile={task_profile}\n".encode("utf-8"))
    digest.update(f"known={task_known}\n".encode("utf-8"))
    digest.update(f"stopat_mode={stopat_mode}\n".encode("utf-8"))
    digest.update(f"blackbox_policy={blackbox_policy}\n".encode("utf-8"))
    digest.update(f"stopats={len(stopats)}\n".encode("utf-8"))
    for stopat in stopats:
        digest.update(stopat.encode("utf-8"))
        digest.update(b"\n")
    return TaskPolicy(
        task_profile=task_profile,
        task_known=task_known,
        stopat_mode=stopat_mode,
        blackbox_policy=blackbox_policy,
        task_policy_fingerprint=digest.hexdigest()[:16],
    )


def contract_fingerprint(
    task: str,
    stopats: tuple[str, ...],
    toplevel: str,
    files: list[str],
    include_dirs: list[str],
    defines: list[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"task={task}\n".encode("utf-8"))
    digest.update(f"stopats={len(stopats)}\n".encode("utf-8"))
    for item in stopats:
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    digest.update(f"toplevel={toplevel}\n".encode("utf-8"))
    digest.update(f"files={len(files)}\n".encode("utf-8"))
    for item in files:
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    digest.update(f"include_dirs={len(include_dirs)}\n".encode("utf-8"))
    for item in include_dirs:
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    digest.update(f"defines={len(defines)}\n".encode("utf-8"))
    for item in defines:
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def run_fusesoc_setup(
    fusesoc_bin: str,
    opentitan_root: Path,
    row: ManifestRow,
    job_dir: Path,
    target: str,
    tool: str,
) -> tuple[int, Path]:
    log_path = job_dir / "fusesoc-setup.log"
    # Some OpenTitan manifests use two-part core names (e.g.
    # "earlgrey_dv:otp_ctrl_sva"). FuseSoC expects a full VLNV, so normalize
    # these legacy names to the OpenTitan vendor namespace.
    core_name = row.fusesoc_core
    if core_name.count(":") == 1:
        core_name = f"lowrisc:{core_name}"
    cmd = [
        fusesoc_bin,
        "--cores-root",
        str(opentitan_root),
        "run",
        "--target",
        target,
        "--tool",
        tool,
        "--setup",
        core_name,
    ]
    proc = subprocess.run(
        cmd,
        cwd=job_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    payload = stdout
    if payload and not payload.endswith("\n"):
        payload += "\n"
    payload += stderr
    log_path.write_text(payload, encoding="utf-8")
    return proc.returncode, log_path


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    opentitan_root = Path(args.opentitan_root).resolve()
    out_contracts = Path(args.out_contracts).resolve()
    if not opentitan_root.is_dir():
        fail(f"opentitan root not found: {opentitan_root}")
    if shutil.which(args.fusesoc_bin) is None and not Path(args.fusesoc_bin).exists():
        fail(f"fusesoc executable not found: {args.fusesoc_bin}")

    rows = read_manifest(manifest_path)
    target_filter = args.target_filter.strip()
    if target_filter:
        try:
            target_re = re.compile(target_filter)
        except re.error as exc:
            fail(f"invalid --target-filter: {target_filter} ({exc})")
        rows = [row for row in rows if target_re.search(row.target_name)]

    user_workdir = Path(args.workdir).resolve() if args.workdir else None
    temp_workdir_obj = None
    if user_workdir is None:
        temp_workdir_obj = tempfile.TemporaryDirectory(
            prefix="opentitan-fpv-contracts-"
        )
        workdir_root = Path(temp_workdir_obj.name)
    else:
        workdir_root = user_workdir
        workdir_root.mkdir(parents=True, exist_ok=True)

    contracts_header = [
        "target_name",
        "fusesoc_core",
        "task",
        "flow",
        "sub_flow",
        "rel_path",
        "toplevel",
        "setup_status",
        "file_count",
        "include_dir_count",
        "define_count",
        "files_fingerprint",
        "include_dirs_fingerprint",
        "defines_fingerprint",
        "contract_fingerprint",
        "stopat_count",
        "stopats_fingerprint",
        "stopats",
        "task_profile",
        "task_known",
        "stopat_mode",
        "blackbox_policy",
        "task_policy_fingerprint",
        "files",
        "include_dirs",
        "defines",
        "eda_yml_path",
        "setup_log_path",
        "workdir",
    ]
    out_contracts.parent.mkdir(parents=True, exist_ok=True)

    errors = 0
    unknown_tasks: set[str] = set()
    with out_contracts.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"#opentitan_compile_contract_schema_version={SCHEMA_VERSION}\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(contracts_header)
        for idx, row in enumerate(rows):
            task_policy = resolve_task_policy(row.task, row.stopats)
            if task_policy.task_known != "1":
                unknown_tasks.add(row.task.strip())
            job_dir = workdir_root / f"{idx:04d}-{sanitize_token(row.target_name)}"
            job_dir.mkdir(parents=True, exist_ok=True)
            rc, log_path = run_fusesoc_setup(
                args.fusesoc_bin,
                opentitan_root,
                row,
                job_dir,
                args.fusesoc_target,
                args.fusesoc_tool,
            )
            eda_path = locate_eda_yml(job_dir)
            setup_status = "ok"
            if eda_path is None:
                setup_status = "error"
                errors += 1
                stopats_fp = hash_lines(list(row.stopats))
                writer.writerow(
                    [
                        row.target_name,
                        row.fusesoc_core,
                        row.task,
                        row.flow,
                        row.sub_flow,
                        row.rel_path,
                        "",
                        setup_status,
                        "0",
                        "0",
                        "0",
                        "",
                        "",
                        "",
                        "",
                        str(len(row.stopats)),
                        stopats_fp,
                        ";".join(row.stopats),
                        task_policy.task_profile,
                        task_policy.task_known,
                        task_policy.stopat_mode,
                        task_policy.blackbox_policy,
                        task_policy.task_policy_fingerprint,
                        "",
                        "",
                        "",
                        "",
                        str(log_path),
                        str(job_dir),
                    ]
                )
                continue
            if rc != 0:
                setup_status = "partial"

            eda_obj = yaml.safe_load(eda_path.read_text(encoding="utf-8")) or {}
            if not isinstance(eda_obj, dict):
                fail(f"invalid eda yml object (not dict): {eda_path}")
            files, include_dirs = resolve_eda_paths(eda_obj.get("files"), eda_path.parent)
            files = apply_platform_file_filter(row.rel_path, files)
            toplevel = resolve_contract_toplevel(
                toplevel_string(eda_obj.get("toplevel")),
                files,
            )
            explicit_incdirs = [
                str((eda_path.parent / item).resolve(strict=False))
                for item in normalize_string_list(eda_obj.get("incdirs"))
            ]
            for incdir in explicit_incdirs:
                if incdir not in include_dirs:
                    include_dirs.append(incdir)
            defines = apply_target_define_overrides(
                row.target_name,
                normalize_string_list(eda_obj.get("vlogdefine")),
            )
            files_fp = hash_lines(files)
            incdirs_fp = hash_lines(include_dirs)
            defines_fp = hash_lines(defines)
            stopats_fp = hash_lines(list(row.stopats))
            contract_fp = contract_fingerprint(
                row.task, row.stopats, toplevel, files, include_dirs, defines
            )
            writer.writerow(
                [
                    row.target_name,
                    row.fusesoc_core,
                    row.task,
                    row.flow,
                    row.sub_flow,
                    row.rel_path,
                    toplevel,
                    setup_status,
                    str(len(files)),
                    str(len(include_dirs)),
                    str(len(defines)),
                    files_fp,
                    incdirs_fp,
                    defines_fp,
                    contract_fp,
                    str(len(row.stopats)),
                    stopats_fp,
                    ";".join(row.stopats),
                    task_policy.task_profile,
                    task_policy.task_known,
                    task_policy.stopat_mode,
                    task_policy.blackbox_policy,
                    task_policy.task_policy_fingerprint,
                    ";".join(files),
                    ";".join(include_dirs),
                    ";".join(defines),
                    str(eda_path),
                    str(log_path),
                    str(job_dir),
                ]
            )

    if temp_workdir_obj is not None and not args.keep_workdir:
        temp_workdir_obj.cleanup()

    if unknown_tasks:
        unknown_rendered = ", ".join(sorted(task for task in unknown_tasks if task))
        if args.fail_on_unknown_task:
            fail(
                "unknown OpenTitan FPV task names detected: "
                + unknown_rendered
                + " (set task profile mapping in resolve_opentitan_formal_compile_contracts.py)"
            )
        print(
            "warning: unknown OpenTitan FPV task names detected: "
            + unknown_rendered,
            file=sys.stderr,
        )

    status = "ok" if errors == 0 else "error"
    print(
        (
            f"resolved opentitan compile contracts: manifest={manifest_path} "
            f"targets={len(rows)} errors={errors} status={status} out={out_contracts}"
        ),
        file=sys.stderr,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Resolve OpenTitan formal cfg targets with dvsim-like --select-cfgs semantics.

This utility expands one or more top-level formal cfg HJSON files and emits a
deterministic target manifest. It supports:
  - recursive `import_cfgs` loading
  - `use_cfgs` expansion (inline dict entries and string references by name)
  - ordered `--select-cfgs` filtering with unknown-target diagnostics
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NamedConfig:
    name: str
    data: dict[str, Any]
    source_path: Path


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def infer_proj_root(cfg_path: Path) -> Path:
    cfg_path = cfg_path.resolve()
    candidates = [cfg_path.parent, *cfg_path.parents]
    for cand in candidates:
        if (cand / "hw").is_dir() and (cand / "util").is_dir():
            return cand
    return cfg_path.parent


def parse_cfg_object(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    hjson_exc: Exception | None = None
    try:
        import hjson  # type: ignore

        obj = hjson.loads(text)
        if not isinstance(obj, dict):
            fail(f"top-level cfg must be an object: {path}")
        return obj
    except Exception as exc:
        hjson_exc = exc

    try:
        obj = json.loads(text)
    except Exception as json_exc:
        fail(
            f"failed to parse cfg '{path}' as hjson/json: "
            f"hjson_error={hjson_exc} json_error={json_exc}"
        )
    if not isinstance(obj, dict):
        fail(f"top-level cfg must be an object: {path}")
    return obj


def resolve_import_path(raw: str, cfg_dir: Path, proj_root: Path) -> Path:
    expanded = raw.replace("{proj_root}", str(proj_root))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = (cfg_dir / candidate).resolve()
    return candidate


def load_cfg_recursive(
    cfg_path: Path,
    proj_root: Path,
    loaded: dict[Path, dict[str, Any]],
    visiting: set[Path],
) -> None:
    cfg_path = cfg_path.resolve()
    if cfg_path in loaded:
        return
    if cfg_path in visiting:
        fail(f"import_cfgs cycle detected at {cfg_path}")
    if not cfg_path.is_file():
        fail(f"cfg file not found: {cfg_path}")

    visiting.add(cfg_path)
    cfg = parse_cfg_object(cfg_path)
    loaded[cfg_path] = cfg

    imports = cfg.get("import_cfgs", [])
    if imports is None:
        imports = []
    if not isinstance(imports, list):
        fail(f"invalid import_cfgs in {cfg_path}: expected list")
    for idx, entry in enumerate(imports):
        if not isinstance(entry, str):
            fail(
                f"invalid import_cfgs entry in {cfg_path}: index={idx} "
                f"expected string, got {type(entry).__name__}"
            )
        import_path = resolve_import_path(entry, cfg_path.parent, proj_root)
        if not import_path.is_file():
            fail(
                f"import_cfgs path not found while loading {cfg_path}: "
                f"{entry} -> {import_path}"
            )
        load_cfg_recursive(import_path, proj_root, loaded, visiting)
    visiting.remove(cfg_path)


def build_named_registry(loaded: dict[Path, dict[str, Any]]) -> dict[str, NamedConfig]:
    registry: dict[str, NamedConfig] = {}
    for path, cfg in loaded.items():
        name = cfg.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        existing = registry.get(name)
        if existing and existing.source_path != path:
            fail(
                "duplicate cfg name in import graph: "
                f"name='{name}' first={existing.source_path} second={path}"
            )
        registry[name] = NamedConfig(name=name, data=cfg, source_path=path)
    return registry


def build_parent_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for key, value in cfg.items():
        if key in ("use_cfgs",):
            continue
        defaults[key] = value
    return defaults


def expand_root_targets(
    root_cfg: dict[str, Any],
    root_cfg_path: Path,
    registry: dict[str, NamedConfig],
) -> list[dict[str, Any]]:
    parent_defaults = build_parent_defaults(root_cfg)
    use_cfgs = root_cfg.get("use_cfgs")
    entries: list[Any]
    if use_cfgs is None:
        entries = [root_cfg]
    else:
        if not isinstance(use_cfgs, list):
            fail(f"invalid use_cfgs in {root_cfg_path}: expected list")
        entries = use_cfgs

    expanded: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        row: dict[str, Any] = dict(parent_defaults)
        source_kind = "inline"
        source_path = root_cfg_path
        if isinstance(entry, dict):
            row.update(entry)
        elif isinstance(entry, str):
            ref_name = entry.strip()
            if not ref_name:
                fail(
                    f"invalid use_cfgs entry in {root_cfg_path}: "
                    f"index={idx} empty reference"
                )
            ref_cfg = registry.get(ref_name)
            if ref_cfg is None:
                available = ", ".join(sorted(registry.keys()))
                fail(
                    f"unknown use_cfgs reference '{ref_name}' in {root_cfg_path}; "
                    f"available imported cfg names: {available}"
                )
            row.update(ref_cfg.data)
            source_kind = "reference"
            source_path = ref_cfg.source_path
        else:
            fail(
                f"invalid use_cfgs entry in {root_cfg_path}: index={idx} "
                f"expected dict|string, got {type(entry).__name__}"
            )

        target_name = row.get("name")
        if not isinstance(target_name, str) or not target_name.strip():
            fail(
                f"expanded target missing non-empty name in {root_cfg_path}: "
                f"index={idx}"
            )
        row["name"] = target_name.strip()
        row["_source_kind"] = source_kind
        row["_source_cfg_file"] = str(source_path)
        row["_root_cfg_file"] = str(root_cfg_path)
        expanded.append(row)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in expanded:
        name = row["name"]
        if name in seen:
            continue
        seen.add(name)
        deduped.append(row)
    return deduped


def target_payload(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith("_"):
            continue
        out[key] = value
    return out


def merge_expanded_targets(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_payloads: dict[str, dict[str, Any]] = {}
    for rows in groups:
        for row in rows:
            name = row["name"]
            payload = target_payload(row)
            existing = seen_payloads.get(name)
            if existing is None:
                seen_payloads[name] = payload
                merged.append(row)
                continue
            if existing != payload:
                fail(
                    "duplicate target name with conflicting payload across cfg files: "
                    f"name='{name}'"
                )
    return merged


def parse_select_cfg_tokens(raw_values: list[str]) -> list[str]:
    tokens: list[str] = []
    for raw in raw_values:
        for token in re.split(r"[,\s]+", raw.strip()):
            if token:
                tokens.append(token)
    ordered_unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered_unique.append(token)
    return ordered_unique


def select_targets(
    expanded_targets: list[dict[str, Any]],
    selected_names: list[str],
) -> list[dict[str, Any]]:
    if not selected_names:
        return expanded_targets
    by_name = {row["name"]: row for row in expanded_targets}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name.keys()))
        fail(
            "unknown --select-cfgs target(s): "
            + ", ".join(missing)
            + f"; available targets: {available}"
        )
    return [by_name[name] for name in selected_names]


def scalar_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    header = [
        "target_name",
        "dut",
        "fusesoc_core",
        "task",
        "stopats",
        "flow",
        "sub_flow",
        "rel_path",
        "source_kind",
        "source_cfg_file",
        "root_cfg_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            out = [
                row.get("name", ""),
                row.get("dut", ""),
                row.get("fusesoc_core", ""),
                row.get("task", ""),
                row.get("stopats", []),
                row.get("flow", ""),
                row.get("sub_flow", ""),
                row.get("rel_path", ""),
                row.get("_source_kind", ""),
                row.get("_source_cfg_file", ""),
                row.get("_root_cfg_file", ""),
            ]
            f.write("\t".join(scalar_str(item) for item in out) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-file",
        action="append",
        required=True,
        help=(
            "OpenTitan formal cfg HJSON file "
            "(repeatable; e.g. top_earlgrey_fpv_*.hjson)"
        ),
    )
    parser.add_argument(
        "--select-cfgs",
        action="append",
        default=[],
        help="Target cfg names to select (repeatable, comma/space separated)",
    )
    parser.add_argument(
        "--out-manifest",
        required=True,
        help="Output TSV manifest path",
    )
    parser.add_argument(
        "--proj-root",
        default="",
        help="Optional OpenTitan project root for {proj_root} expansion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_paths = [Path(item).resolve() for item in args.cfg_file]
    out_path = Path(args.out_manifest).resolve()
    if args.proj_root:
        proj_root = Path(args.proj_root).resolve()
    else:
        proj_root = infer_proj_root(cfg_paths[0])

    loaded: dict[Path, dict[str, Any]] = {}
    for cfg_path in cfg_paths:
        load_cfg_recursive(cfg_path, proj_root, loaded, set())
    registry = build_named_registry(loaded)
    expanded_targets = merge_expanded_targets(
        [
            expand_root_targets(loaded[cfg_path], cfg_path, registry)
            for cfg_path in cfg_paths
        ]
    )
    selected_names = parse_select_cfg_tokens(args.select_cfgs)
    selected_targets = select_targets(expanded_targets, selected_names)
    write_manifest(out_path, selected_targets)

    cfg_files_rendered = ",".join(str(path) for path in cfg_paths)
    print(
        "opentitan cfg manifest: "
        f"cfg_count={len(cfg_paths)} cfg_files={cfg_files_rendered} "
        f"targets={len(selected_targets)} out={out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

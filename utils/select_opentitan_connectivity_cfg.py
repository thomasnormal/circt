#!/usr/bin/env python3
"""Resolve OpenTitan connectivity cfg and emit normalized connectivity manifests.

This utility ingests an OpenTitan connectivity cfg HJSON (for example
`chip_conn_cfg.hjson`) and emits:
  - target manifest (one normalized target row)
  - rule manifest (normalized CONNECTION/CONDITION rows from CSV inputs)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")


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


def build_cfg_variables(cfg: dict[str, Any], proj_root: Path) -> dict[str, str]:
    variables: dict[str, str] = {"proj_root": str(proj_root)}
    for _ in range(16):
        changed = False
        for key, value in cfg.items():
            if key == "import_cfgs":
                continue
            if not isinstance(value, str):
                continue
            expanded = substitute_tokens(value, variables)
            if variables.get(key) != expanded:
                variables[key] = expanded
                changed = True
        if not changed:
            break
    return variables


def resolve_import_path(
    raw: str,
    cfg_dir: Path,
    variables: dict[str, str],
    label: str,
) -> Path:
    expanded = expand_required(raw, variables, label)
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
    local_variables = build_cfg_variables(cfg, proj_root)

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
        import_path = resolve_import_path(
            entry,
            cfg_path.parent,
            local_variables,
            f"{cfg_path}:import_cfgs[{idx}]",
        )
        if not import_path.is_file():
            fail(
                f"import_cfgs path not found while loading {cfg_path}: "
                f"{entry} -> {import_path}"
            )
        load_cfg_recursive(import_path, proj_root, loaded, visiting)
    visiting.remove(cfg_path)


def compose_effective_cfg(
    cfg_path: Path,
    proj_root: Path,
    loaded: dict[Path, dict[str, Any]],
    composed: dict[Path, dict[str, Any]],
    composing: set[Path],
) -> dict[str, Any]:
    cfg_path = cfg_path.resolve()
    cached = composed.get(cfg_path)
    if cached is not None:
        return dict(cached)
    if cfg_path in composing:
        fail(f"import_cfgs cycle detected while composing cfg: {cfg_path}")
    composing.add(cfg_path)
    cfg = loaded[cfg_path]
    local_variables = build_cfg_variables(cfg, proj_root)
    merged: dict[str, Any] = {}
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
        import_path = resolve_import_path(
            entry,
            cfg_path.parent,
            local_variables,
            f"{cfg_path}:import_cfgs[{idx}]",
        )
        merged.update(
            compose_effective_cfg(
                import_path, proj_root, loaded, composed, composing
            )
        )
    for key, value in cfg.items():
        if key == "import_cfgs":
            continue
        merged[key] = value
    composing.remove(cfg_path)
    composed[cfg_path] = dict(merged)
    return merged


def substitute_tokens(value: str, variables: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return variables.get(key, match.group(0))

    return TOKEN_RE.sub(repl, value)


def expand_required(value: str, variables: dict[str, str], label: str) -> str:
    expanded = value
    for _ in range(16):
        next_value = substitute_tokens(expanded, variables)
        if next_value == expanded:
            break
        expanded = next_value
    unresolved = sorted(set(TOKEN_RE.findall(expanded)))
    if unresolved:
        fail(
            f"unresolved placeholder(s) in {label}: "
            + ", ".join(unresolved)
            + f" (value='{value}')"
        )
    return expanded


def build_variable_map(effective_cfg: dict[str, Any], proj_root: Path) -> dict[str, str]:
    variables: dict[str, str] = {"proj_root": str(proj_root)}
    for _ in range(16):
        changed = False
        for key, value in effective_cfg.items():
            if not isinstance(value, str):
                continue
            expanded = substitute_tokens(value, variables)
            if variables.get(key) != expanded:
                variables[key] = expanded
                changed = True
        if not changed:
            break
    return variables


def parse_conn_csvs(
    cfg_path: Path,
    effective_cfg: dict[str, Any],
    variables: dict[str, str],
) -> list[Path]:
    raw_conn_csvs = effective_cfg.get("conn_csvs")
    if raw_conn_csvs is None:
        fail(f"connectivity cfg missing conn_csvs list: {cfg_path}")
    if not isinstance(raw_conn_csvs, list):
        fail(f"invalid conn_csvs in {cfg_path}: expected list")

    csv_paths: list[Path] = []
    for idx, entry in enumerate(raw_conn_csvs):
        if not isinstance(entry, str):
            fail(
                f"invalid conn_csvs entry in {cfg_path}: index={idx} "
                f"expected string, got {type(entry).__name__}"
            )
        expanded = expand_required(entry, variables, f"{cfg_path}:conn_csvs[{idx}]")
        csv_path = Path(expanded)
        if not csv_path.is_absolute():
            csv_path = (cfg_path.parent / csv_path).resolve()
        if not csv_path.is_file():
            fail(
                f"connectivity CSV path not found while loading {cfg_path}: "
                f"{entry} -> {csv_path}"
            )
        csv_paths.append(csv_path)
    return csv_paths


def parse_csv_rules(csv_path: Path) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    condition_seq = 0
    seen_rule_ids: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue
            first = (row[0] if len(row) > 0 else "").strip()
            if not first:
                continue
            if first.startswith("#"):
                continue
            kind = first.upper()
            if kind not in {"CONNECTION", "CONDITION"}:
                fail(
                    f"unsupported connectivity CSV row kind '{first}' in "
                    f"{csv_path}:{line_no}; expected CONNECTION or CONDITION"
                )

            cols = list(row[:6])
            if len(cols) < 6:
                cols.extend([""] * (6 - len(cols)))
            if kind == "CONNECTION":
                rule_name = cols[1].strip()
                if not rule_name:
                    fail(
                        f"connectivity CONNECTION row missing NAME in "
                        f"{csv_path}:{line_no}"
                    )
                src_block = cols[2].strip()
                src_signal = cols[3].strip()
                dest_block = cols[4].strip()
                dest_signal = cols[5].strip()
            else:
                condition_seq += 1
                rule_name = f"CONDITION_{condition_seq}"
                # CONDITION rows in OpenTitan connectivity CSVs do not carry a
                # NAME column. They follow:
                # CONDITION, SRC BLOCK, SRC SIGNAL, EXPECTED_TRUE, EXPECTED_FALSE
                src_block = cols[1].strip()
                src_signal = cols[2].strip()
                dest_block = cols[3].strip()
                dest_signal = cols[4].strip()

            rule_id = f"{csv_path.name}:{rule_name}"
            if rule_id in seen_rule_ids:
                fail(f"duplicate connectivity rule_id '{rule_id}' in {csv_path}")
            seen_rule_ids.add(rule_id)
            rules.append(
                {
                    "rule_id": rule_id,
                    "rule_type": kind,
                    "csv_file": str(csv_path),
                    "csv_row": line_no,
                    "rule_name": rule_name,
                    "src_block": src_block,
                    "src_signal": src_signal,
                    "dest_block": dest_block,
                    "dest_signal": dest_signal,
                }
            )
    return rules


def scalar_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_target_manifest(path: Path, row: dict[str, Any]) -> None:
    header = [
        "target_name",
        "flow",
        "sub_flow",
        "fusesoc_core",
        "rel_path",
        "bbox_cmd",
        "conn_csv_count",
        "conn_csvs",
        "cfg_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        out = [row.get(key, "") for key in header]
        f.write("\t".join(scalar_str(item) for item in out) + "\n")


def write_rules_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    header = [
        "rule_id",
        "rule_type",
        "csv_file",
        "csv_row",
        "rule_name",
        "src_block",
        "src_signal",
        "dest_block",
        "dest_signal",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            out = [row.get(key, "") for key in header]
            f.write("\t".join(scalar_str(item) for item in out) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-file",
        required=True,
        help="OpenTitan connectivity cfg HJSON file (e.g. chip_conn_cfg.hjson)",
    )
    parser.add_argument(
        "--out-target-manifest",
        required=True,
        help="Output TSV path for normalized connectivity target manifest",
    )
    parser.add_argument(
        "--out-rules-manifest",
        required=True,
        help="Output TSV path for normalized connectivity rules manifest",
    )
    parser.add_argument(
        "--proj-root",
        default="",
        help="Optional OpenTitan project root for {proj_root} expansion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.cfg_file).resolve()
    out_target_manifest = Path(args.out_target_manifest).resolve()
    out_rules_manifest = Path(args.out_rules_manifest).resolve()
    if args.proj_root:
        proj_root = Path(args.proj_root).resolve()
    else:
        proj_root = infer_proj_root(cfg_path)

    loaded: dict[Path, dict[str, Any]] = {}
    load_cfg_recursive(cfg_path, proj_root, loaded, set())
    effective_cfg = compose_effective_cfg(cfg_path, proj_root, loaded, {}, set())

    target_name = str(effective_cfg.get("name", "")).strip()
    if not target_name:
        fail(f"connectivity cfg missing non-empty name: {cfg_path}")
    fusesoc_core = str(effective_cfg.get("fusesoc_core", "")).strip()
    if not fusesoc_core:
        fail(f"connectivity cfg missing non-empty fusesoc_core: {cfg_path}")

    variables = build_variable_map(effective_cfg, proj_root)
    csv_paths = parse_conn_csvs(cfg_path, effective_cfg, variables)
    rule_rows: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        rule_rows.extend(parse_csv_rules(csv_path))

    target_row = {
        "target_name": target_name,
        "flow": str(effective_cfg.get("flow", "")).strip(),
        "sub_flow": str(effective_cfg.get("sub_flow", "")).strip(),
        "fusesoc_core": fusesoc_core,
        "rel_path": str(effective_cfg.get("rel_path", "")).strip(),
        "bbox_cmd": str(effective_cfg.get("bbox_cmd", "")).strip(),
        "conn_csv_count": len(csv_paths),
        "conn_csvs": [str(path) for path in csv_paths],
        "cfg_file": str(cfg_path),
    }

    write_target_manifest(out_target_manifest, target_row)
    write_rules_manifest(out_rules_manifest, rule_rows)

    print(
        "opentitan connectivity manifest: "
        f"cfg={cfg_path} csv_files={len(csv_paths)} rules={len(rule_rows)} "
        f"out_target={out_target_manifest} out_rules={out_rules_manifest}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Write a WS0 baseline manifest for OpenTitan/sv-tests formal lanes."""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path


def quote(arg: str) -> str:
    return shlex.quote(arg)


def parse_env_assignments(raw_entries: list[str], lane: str) -> list[str]:
    assignments: list[str] = []
    for raw in raw_entries:
        token = raw.strip()
        if not token:
            continue
        if "=" not in token:
            raise SystemExit(
                f"invalid {lane} env entry (expected KEY=VALUE): {raw}"
            )
        key, _value = token.split("=", 1)
        if not key.strip():
            raise SystemExit(
                f"invalid {lane} env entry (empty key): {raw}"
            )
        assignments.append(token)
    return assignments


def build_command_with_extras(
    base: list[str], extras: list[str], env_assignments: list[str]
) -> str:
    parts = [quote(token) for token in base]
    if env_assignments:
        parts = ["env", *[quote(token) for token in env_assignments], *parts]
    for extra in extras:
        token = extra.strip()
        if token:
            parts.append(token)
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write WS0 baseline manifest for formal runner lanes."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-id", default="ws0-formal-baseline")
    parser.add_argument(
        "--generated-at",
        default="",
        help="Override generated timestamp (RFC3339 UTC).",
    )
    parser.add_argument(
        "--opentitan-root",
        default=str(Path.home() / "opentitan"),
    )
    parser.add_argument(
        "--sv-tests-root",
        default=str(Path.home() / "sv-tests"),
    )
    parser.add_argument(
        "--connectivity-target-manifest",
        default="",
        help="Optional connectivity target manifest; requires rules manifest.",
    )
    parser.add_argument(
        "--connectivity-rules-manifest",
        default="",
        help="Optional connectivity rules manifest; requires target manifest.",
    )
    parser.add_argument(
        "--aes-extra",
        action="append",
        default=[],
        help="Extra shell tokens appended to the AES LEC command.",
    )
    parser.add_argument(
        "--aes-env",
        action="append",
        default=[],
        help="Environment assignment (KEY=VALUE) for AES LEC command.",
    )
    parser.add_argument(
        "--connectivity-extra",
        action="append",
        default=[],
        help="Extra shell tokens appended to the connectivity LEC command.",
    )
    parser.add_argument(
        "--connectivity-env",
        action="append",
        default=[],
        help="Environment assignment (KEY=VALUE) for connectivity LEC command.",
    )
    parser.add_argument(
        "--bmc-extra",
        action="append",
        default=[],
        help="Extra shell tokens appended to the sv-tests BMC command.",
    )
    parser.add_argument(
        "--bmc-env",
        action="append",
        default=[],
        help="Environment assignment (KEY=VALUE) for sv-tests BMC command.",
    )
    args = parser.parse_args()

    generated_at = args.generated_at.strip()
    if not generated_at:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    target_manifest = args.connectivity_target_manifest.strip()
    rules_manifest = args.connectivity_rules_manifest.strip()
    if bool(target_manifest) != bool(rules_manifest):
        raise SystemExit(
            "connectivity manifest options must be provided together: "
            "--connectivity-target-manifest and --connectivity-rules-manifest"
        )
    aes_env = parse_env_assignments(args.aes_env, "aes")
    connectivity_env = parse_env_assignments(args.connectivity_env, "connectivity")
    bmc_env = parse_env_assignments(args.bmc_env, "bmc")
    repo_root = Path(__file__).resolve().parents[2]

    commands: list[dict[str, str]] = []
    commands.append(
        {
            "id": "ws0_aes_lec",
            "suite": "opentitan",
            "mode": "LEC",
            "cwd": str(repo_root),
            "command": build_command_with_extras(
                [
                    "utils/run_opentitan_circt_lec.py",
                    "--opentitan-root",
                    str(Path(args.opentitan_root).expanduser()),
                ],
                args.aes_extra,
                aes_env,
            ),
        }
    )
    if target_manifest and rules_manifest:
        commands.append(
            {
                "id": "ws0_connectivity_lec",
                "suite": "opentitan",
                "mode": "CONNECTIVITY_LEC",
                "cwd": str(repo_root),
                "command": build_command_with_extras(
                    [
                        "utils/run_opentitan_connectivity_circt_lec.py",
                        "--target-manifest",
                        str(Path(target_manifest).expanduser()),
                        "--rules-manifest",
                        str(Path(rules_manifest).expanduser()),
                        "--opentitan-root",
                        str(Path(args.opentitan_root).expanduser()),
                    ],
                    args.connectivity_extra,
                    connectivity_env,
                ),
            }
        )
    commands.append(
        {
            "id": "ws0_sv_tests_bmc",
            "suite": "sv-tests",
            "mode": "BMC",
            "cwd": str(repo_root),
            "command": build_command_with_extras(
                [
                    "utils/run_sv_tests_circt_bmc.sh",
                    str(Path(args.sv_tests_root).expanduser()),
                ],
                args.bmc_extra,
                bmc_env,
            ),
        }
    )

    payload = {
        "schema_version": 1,
        "baseline_id": args.baseline_id,
        "generated_at": generated_at,
        "metadata": {
            "ws": "WS0",
            "opentitan_root": str(Path(args.opentitan_root).expanduser()),
            "sv_tests_root": str(Path(args.sv_tests_root).expanduser()),
        },
        "commands": commands,
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def build_command_with_extras(base: list[str], extras: list[str]) -> str:
    parts = [quote(token) for token in base]
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
        "--connectivity-extra",
        action="append",
        default=[],
        help="Extra shell tokens appended to the connectivity LEC command.",
    )
    parser.add_argument(
        "--bmc-extra",
        action="append",
        default=[],
        help="Extra shell tokens appended to the sv-tests BMC command.",
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

    commands: list[dict[str, str]] = []
    commands.append(
        {
            "id": "ws0_aes_lec",
            "suite": "opentitan",
            "mode": "LEC",
            "command": build_command_with_extras(
                [
                    "utils/run_opentitan_circt_lec.py",
                    "--opentitan-root",
                    str(Path(args.opentitan_root).expanduser()),
                ],
                args.aes_extra,
            ),
        }
    )
    if target_manifest and rules_manifest:
        commands.append(
            {
                "id": "ws0_connectivity_lec",
                "suite": "opentitan",
                "mode": "CONNECTIVITY_LEC",
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
                ),
            }
        )
    commands.append(
        {
            "id": "ws0_sv_tests_bmc",
            "suite": "sv-tests",
            "mode": "BMC",
            "command": build_command_with_extras(
                [
                    "utils/run_sv_tests_circt_bmc.sh",
                    str(Path(args.sv_tests_root).expanduser()),
                ],
                args.bmc_extra,
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

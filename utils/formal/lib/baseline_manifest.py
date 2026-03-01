#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared baseline manifest parser/validator helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ManifestValidationError(ValueError):
    """Raised when baseline manifest payloads fail schema checks."""


@dataclass(frozen=True)
class ManifestCommand:
    command_index: int
    suite: str
    mode: str
    case_label: str
    command: str
    cwd: str
    timeout_secs: int
    expected_returncodes: tuple[int, ...]


def slugify(token: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9._-]+", "_", token.strip())
    compact = compact.strip("._-")
    return compact or "command"


def parse_timeout_value(raw: object, field_name: str, index: int) -> int:
    if raw is None or raw == "":
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(
            f"manifest commands[{index}] {field_name} must be an integer "
            f"(got: {raw!r})"
        ) from exc
    if value < 0:
        raise ManifestValidationError(
            f"manifest commands[{index}] {field_name} must be non-negative "
            f"(got: {raw!r})"
        )
    return value


def parse_expected_returncodes(
    raw: object, field_name: str, index: int
) -> tuple[int, ...]:
    tokens: list[object]
    if raw is None or raw == "":
        return (0,)
    if isinstance(raw, int):
        tokens = [raw]
    elif isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        tokens = raw
    else:
        raise ManifestValidationError(
            f"manifest commands[{index}] {field_name} must be an integer, "
            "comma-separated string, or list of integers"
        )
    if not tokens:
        raise ManifestValidationError(
            f"manifest commands[{index}] {field_name} must be non-empty"
        )
    out: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        try:
            code = int(token)
        except (TypeError, ValueError) as exc:
            raise ManifestValidationError(
                f"manifest commands[{index}] {field_name} contains "
                f"non-integer value: {token!r}"
            ) from exc
        if code not in seen:
            seen.add(code)
            out.append(code)
    return tuple(out)


def _expect_nonempty_string(
    *,
    value: object,
    context: str,
) -> str:
    if not isinstance(value, str):
        raise ManifestValidationError(f"{context} must be a string")
    token = value.strip()
    if not token:
        raise ManifestValidationError(f"{context} must be non-empty")
    return token


def _expect_optional_string(
    *,
    value: object,
    context: str,
) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ManifestValidationError(f"{context} must be a string")
    return value.strip()


def _validate_manifest_root(payload: dict[str, Any]) -> None:
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ManifestValidationError("manifest schema_version must be 1")
    _expect_nonempty_string(value=payload.get("baseline_id"), context="baseline_id")
    _expect_nonempty_string(value=payload.get("generated_at"), context="generated_at")
    metadata = payload.get("metadata", {})
    if metadata is not None and not isinstance(metadata, dict):
        raise ManifestValidationError("manifest metadata must be an object")


def load_manifest_commands(path: Path) -> tuple[dict[str, Any], list[ManifestCommand]]:
    if not path.is_file():
        raise ManifestValidationError(f"manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"invalid manifest JSON ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestValidationError(
            f"invalid manifest root (expected object): {path}"
        )

    _validate_manifest_root(payload)
    raw_commands = payload.get("commands", [])
    if not isinstance(raw_commands, list) or not raw_commands:
        raise ManifestValidationError("manifest missing non-empty commands list")

    commands: list[ManifestCommand] = []
    seen_case_labels: set[str] = set()
    for index, raw in enumerate(raw_commands, start=1):
        if not isinstance(raw, dict):
            raise ManifestValidationError(f"manifest commands[{index}] must be an object")
        suite = _expect_nonempty_string(
            value=raw.get("suite"), context=f"manifest commands[{index}] suite"
        )
        mode = _expect_nonempty_string(
            value=raw.get("mode"), context=f"manifest commands[{index}] mode"
        )
        command = _expect_nonempty_string(
            value=raw.get("command"), context=f"manifest commands[{index}] command"
        )
        label_seed = (
            _expect_optional_string(
                value=raw.get("id", ""), context=f"manifest commands[{index}] id"
            )
            or _expect_optional_string(
                value=raw.get("label", ""),
                context=f"manifest commands[{index}] label",
            )
            or f"{suite}_{mode}_{index}"
        )
        case_label = slugify(label_seed)
        if case_label in seen_case_labels:
            raise ManifestValidationError(
                "manifest has duplicate command case label after slugify: "
                f"{case_label} (command index {index})"
            )
        seen_case_labels.add(case_label)

        commands.append(
            ManifestCommand(
                command_index=index,
                suite=suite,
                mode=mode,
                case_label=case_label,
                command=command,
                cwd=_expect_optional_string(
                    value=raw.get("cwd", ""),
                    context=f"manifest commands[{index}] cwd",
                ),
                timeout_secs=parse_timeout_value(
                    raw.get("timeout_secs", 0), "timeout_secs", index
                ),
                expected_returncodes=parse_expected_returncodes(
                    raw.get("expected_returncodes", 0),
                    "expected_returncodes",
                    index,
                ),
            )
        )
    return payload, commands

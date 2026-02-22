#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared helpers for formal runner scripts."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

Allowlist = tuple[set[str], list[str], list[re.Pattern[str]]]
DriftRow = tuple[str, str, str, str, str]


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def parse_nonnegative_int(
    raw: str, name: str, fail_fn: Callable[[str], None] | None = None
) -> int:
    if fail_fn is None:
        fail_fn = fail
    try:
        value = int(raw)
    except ValueError:
        fail_fn(f"invalid {name}: {raw}")
        raise AssertionError("unreachable")
    if value < 0:
        fail_fn(f"invalid {name}: {raw}")
        raise AssertionError("unreachable")
    return value


def parse_nonnegative_float(
    raw: str, name: str, fail_fn: Callable[[str], None] | None = None
) -> float:
    if fail_fn is None:
        fail_fn = fail
    try:
        value = float(raw)
    except ValueError:
        fail_fn(f"invalid {name}: {raw}")
        raise AssertionError("unreachable")
    if value < 0.0:
        fail_fn(f"invalid {name}: {raw}")
        raise AssertionError("unreachable")
    return value


def parse_exit_codes(
    raw: str, name: str, fail_fn: Callable[[str], None] | None = None
) -> set[int]:
    if fail_fn is None:
        fail_fn = fail
    text = raw.strip()
    if not text:
        return set()
    out: set[int] = set()
    for token in text.split(","):
        code_text = token.strip()
        if not code_text:
            continue
        try:
            out.add(int(code_text))
        except ValueError:
            fail_fn(f"invalid {name}: {raw}")
            raise AssertionError("unreachable")
    return out


def load_allowlist(
    path: Path, fail_fn: Callable[[str], None] | None = None
) -> Allowlist:
    if fail_fn is None:
        fail_fn = fail
    exact: set[str] = set()
    prefixes: list[str] = []
    regex_rules: list[re.Pattern[str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            mode = "exact"
            payload = line
            if ":" in line:
                mode, payload = line.split(":", 1)
                mode = mode.strip()
                payload = payload.strip()
            if not payload:
                fail_fn(f"invalid allowlist row {line_no}: empty pattern")
                raise AssertionError("unreachable")
            if mode == "exact":
                exact.add(payload)
            elif mode == "prefix":
                prefixes.append(payload)
            elif mode == "regex":
                try:
                    regex_rules.append(re.compile(payload))
                except re.error as exc:
                    fail_fn(
                        f"invalid allowlist row {line_no}: bad regex '{payload}': {exc}"
                    )
                    raise AssertionError("unreachable")
            else:
                fail_fn(
                    f"invalid allowlist row {line_no}: unsupported mode '{mode}' "
                    "(expected exact|prefix|regex)"
                )
                raise AssertionError("unreachable")
    return exact, prefixes, regex_rules


def is_allowlisted(token: str, allowlist: Allowlist) -> bool:
    exact, prefixes, regex_rules = allowlist
    if token in exact:
        return True
    for prefix in prefixes:
        if token.startswith(prefix):
            return True
    for pattern in regex_rules:
        if pattern.search(token):
            return True
    return False


def write_log(path: Path, stdout: str, stderr: str) -> None:
    payload = ""
    if stdout:
        payload += stdout
        if not payload.endswith("\n"):
            payload += "\n"
    if stderr:
        payload += stderr
    path.write_text(payload, encoding="utf-8")


def run_command_logged(
    cmd: list[str],
    log_path: Path,
    *,
    timeout_secs: int = 0,
    out_path: Path | None = None,
    retry_attempts: int = 1,
    retry_backoff_secs: float = 0.0,
    retryable_exit_codes: set[int] | None = None,
    retryable_output_patterns: Sequence[str] = (),
) -> str:
    attempts = retry_attempts if retry_attempts > 0 else 1
    retry_codes = retryable_exit_codes if retryable_exit_codes is not None else set()
    output_patterns = tuple(p.lower() for p in retryable_output_patterns if p)
    last_stdout = ""
    last_stderr = ""
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_secs if timeout_secs > 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            last_stdout = exc.stdout or ""
            last_stderr = exc.stderr or ""
            write_log(log_path, last_stdout, last_stderr)
            if out_path is not None:
                out_path.write_text(last_stdout, encoding="utf-8")
            if attempt < attempts:
                if retry_backoff_secs > 0.0:
                    time.sleep(retry_backoff_secs)
                continue
            raise

        last_stdout = result.stdout or ""
        last_stderr = result.stderr or ""
        write_log(log_path, last_stdout, last_stderr)
        if out_path is not None:
            out_path.write_text(last_stdout, encoding="utf-8")
        if result.returncode == 0:
            return last_stdout + "\n" + last_stderr

        retryable = result.returncode in retry_codes
        if not retryable and output_patterns:
            combined = f"{last_stdout}\n{last_stderr}".lower()
            retryable = any(pattern in combined for pattern in output_patterns)
        if retryable and attempt < attempts:
            if retry_backoff_secs > 0.0:
                time.sleep(retry_backoff_secs)
            continue
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    raise AssertionError("unreachable")


def write_status_summary(
    path: Path,
    fields: Sequence[str],
    by_key: dict[str, dict[str, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["rule_id", *fields])
        for rule_id in sorted(by_key.keys()):
            counts = by_key[rule_id]
            writer.writerow([rule_id, *[counts[field] for field in fields]])


def read_status_summary(
    path: Path,
    fields: Sequence[str],
    label: str,
    fail_fn: Callable[[str], None] | None = None,
) -> dict[str, dict[str, str]]:
    if fail_fn is None:
        fail_fn = fail
    if not path.is_file():
        fail_fn(f"{label} file not found: {path}")
        raise AssertionError("unreachable")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail_fn(f"{label} missing header row: {path}")
            raise AssertionError("unreachable")
        required = {"rule_id", *fields}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            fail_fn(
                f"{label} missing required columns {missing}: "
                f"{path} (found: {reader.fieldnames})"
            )
            raise AssertionError("unreachable")
        out: dict[str, dict[str, str]] = {}
        for idx, row in enumerate(reader, start=2):
            rule_id = (row.get("rule_id") or "").strip()
            if not rule_id:
                continue
            if rule_id in out:
                fail_fn(f"duplicate rule_id '{rule_id}' in {path} row {idx}")
                raise AssertionError("unreachable")
            out[rule_id] = {field: (row.get(field) or "").strip() for field in fields}
    return out


def write_status_drift(path: Path, rows: Iterable[DriftRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["rule_id", "kind", "baseline", "current", "allowlisted"])
        for row in rows:
            writer.writerow(row)


def compute_status_drift(
    baseline: dict[str, dict[str, str]],
    current: dict[str, dict[str, str]],
    fields: Sequence[str],
    allowlist: Allowlist,
) -> tuple[list[DriftRow], list[DriftRow]]:
    drift_rows: list[DriftRow] = []
    non_allowlisted_rows: list[DriftRow] = []

    def add_drift(rule_id: str, kind: str, before: str, after: str) -> None:
        allowlisted = is_allowlisted(rule_id, allowlist)
        row = (rule_id, kind, before, after, "1" if allowlisted else "0")
        drift_rows.append(row)
        if not allowlisted:
            non_allowlisted_rows.append(row)

    baseline_rules = set(baseline.keys())
    current_rules = set(current.keys())
    for rule_id in sorted(baseline_rules - current_rules):
        add_drift(rule_id, "missing_in_current", "present", "absent")
    for rule_id in sorted(current_rules - baseline_rules):
        add_drift(rule_id, "new_in_current", "absent", "present")
    for rule_id in sorted(baseline_rules.intersection(current_rules)):
        before_row = baseline[rule_id]
        after_row = current[rule_id]
        for kind in fields:
            before = before_row.get(kind, "")
            after = after_row.get(kind, "")
            if before != after:
                add_drift(rule_id, kind, before, after)
    return drift_rows, non_allowlisted_rows

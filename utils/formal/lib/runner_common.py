#!/usr/bin/env python3
# Copyright 2026 The CIRCT Authors.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared helpers for formal runner scripts."""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

Allowlist = tuple[set[str], list[str], list[re.Pattern[str]]]
DriftRow = tuple[str, str, str, str, str]
DEFAULT_RETRYABLE_PATTERNS = (
    "text file busy",
    "resource temporarily unavailable",
    "stale file handle",
)


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


def parse_retryable_patterns(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def normalize_drop_reason(line: str) -> str:
    line = line.replace("\t", " ").strip()
    line = re.sub(r"^[^:\n]+:\d+(?::\d+)?:\s*", "", line)
    line = re.sub(r"^[Ww]arning:\s*", "", line)
    line = re.sub(r"\s+", " ", line)
    line = re.sub(r"\d+", "<n>", line)
    line = line.replace(";", ",")
    if len(line) > 240:
        line = line[:240]
    return line


def extract_drop_reasons(log_text: str, pattern: str) -> list[str]:
    reasons: set[str] = set()
    for line in log_text.splitlines():
        if pattern not in line:
            continue
        reason = normalize_drop_reason(line)
        if reason:
            reasons.add(reason)
    return sorted(reasons)


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


def _coerce_text(payload: str | bytes | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return payload


def _truncate_log_bytes(
    encoded: bytes, max_log_bytes: int, truncation_label: str
) -> bytes:
    if max_log_bytes <= 0 or len(encoded) <= max_log_bytes:
        return encoded
    notice = (
        f"\n[{truncation_label}] log truncated from "
        f"{len(encoded)} to {max_log_bytes} bytes\n"
    ).encode("utf-8")
    if max_log_bytes <= len(notice):
        return encoded[:max_log_bytes]
    keep = max_log_bytes - len(notice)
    return encoded[:keep] + notice


def write_log(
    path: Path,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    *,
    max_log_bytes: int = 0,
    truncation_label: str = "formal_runner_common",
) -> None:
    stdout_text = _coerce_text(stdout)
    stderr_text = _coerce_text(stderr)
    payload = ""
    if stdout_text:
        payload += stdout_text
        if not payload.endswith("\n"):
            payload += "\n"
    if stderr_text:
        payload += stderr_text
    encoded = payload.encode("utf-8", errors="replace")
    path.write_bytes(_truncate_log_bytes(encoded, max_log_bytes, truncation_label))


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
    max_log_bytes: int = 0,
    truncation_label: str = "formal_runner_common",
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
            last_stdout = _coerce_text(exc.stdout)
            last_stderr = _coerce_text(exc.stderr)
            write_log(
                log_path,
                last_stdout,
                last_stderr,
                max_log_bytes=max_log_bytes,
                truncation_label=truncation_label,
            )
            if out_path is not None:
                out_path.write_text(last_stdout, encoding="utf-8")
            if attempt < attempts:
                if retry_backoff_secs > 0.0:
                    time.sleep(retry_backoff_secs)
                continue
            raise
        except OSError as exc:
            last_stdout = ""
            last_stderr = f"{exc.__class__.__name__}: {exc}"
            write_log(
                log_path,
                last_stdout,
                last_stderr,
                max_log_bytes=max_log_bytes,
                truncation_label=truncation_label,
            )
            if out_path is not None:
                out_path.write_text(last_stdout, encoding="utf-8")
            retryable = 127 in retry_codes
            if not retryable and output_patterns:
                retryable = any(pattern in last_stderr.lower() for pattern in output_patterns)
            if retryable and attempt < attempts:
                if retry_backoff_secs > 0.0:
                    time.sleep(retry_backoff_secs)
                continue
            raise subprocess.CalledProcessError(
                127,
                cmd,
                output=last_stdout,
                stderr=last_stderr,
            ) from exc

        last_stdout = result.stdout or ""
        last_stderr = result.stderr or ""
        write_log(
            log_path,
            last_stdout,
            last_stderr,
            max_log_bytes=max_log_bytes,
            truncation_label=truncation_label,
        )
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


def run_command_logged_with_env_retry(
    cmd: list[str],
    log_path: Path,
    *,
    timeout_secs: int = 0,
    out_path: Path | None = None,
    fail_fn: Callable[[str], None] | None = None,
    env: dict[str, str] | None = None,
    max_log_bytes: int = 0,
    truncation_label: str = "formal_runner_common",
) -> str:
    if fail_fn is None:
        fail_fn = fail
    resolved_env = os.environ if env is None else env
    retry_attempts = parse_nonnegative_int(
        resolved_env.get("FORMAL_LAUNCH_RETRY_ATTEMPTS", "1"),
        "FORMAL_LAUNCH_RETRY_ATTEMPTS",
        fail_fn,
    )
    retry_backoff_secs = parse_nonnegative_float(
        resolved_env.get("FORMAL_LAUNCH_RETRY_BACKOFF_SECS", "0.2"),
        "FORMAL_LAUNCH_RETRY_BACKOFF_SECS",
        fail_fn,
    )
    retryable_exit_codes = parse_exit_codes(
        resolved_env.get("FORMAL_LAUNCH_RETRYABLE_EXIT_CODES", "126,127"),
        "FORMAL_LAUNCH_RETRYABLE_EXIT_CODES",
        fail_fn,
    )
    retryable_patterns = parse_retryable_patterns(
        resolved_env.get(
            "FORMAL_LAUNCH_RETRYABLE_PATTERNS",
            ",".join(DEFAULT_RETRYABLE_PATTERNS),
        )
    )
    return run_command_logged(
        cmd,
        log_path,
        timeout_secs=timeout_secs,
        out_path=out_path,
        retry_attempts=retry_attempts,
        retry_backoff_secs=retry_backoff_secs,
        retryable_exit_codes=retryable_exit_codes,
        retryable_output_patterns=retryable_patterns,
        max_log_bytes=max_log_bytes,
        truncation_label=truncation_label,
    )


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

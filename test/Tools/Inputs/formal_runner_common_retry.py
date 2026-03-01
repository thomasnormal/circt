#!/usr/bin/env python3
"""Exercise shared formal runner retry helper behavior."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: formal_runner_common_retry.py <helper_dir> <tmp_dir>", file=sys.stderr)
        return 2

    helper_dir = Path(sys.argv[1]).resolve()
    tmp_dir = Path(sys.argv[2]).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(helper_dir))
    import runner_common

    marker = tmp_dir / "retry.marker"
    log_path = tmp_dir / "runner.log"
    out_path = tmp_dir / "runner.out"

    cmd_payload = "\n".join(
        [
            "import pathlib",
            "import sys",
            f"marker = pathlib.Path({str(marker)!r})",
            "if not marker.exists():",
            "    marker.write_text('1', encoding='utf-8')",
            "    print('retry-me')",
            "    raise SystemExit(126)",
            "print('ok-retry')",
        ]
    )

    output = runner_common.run_command_logged(
        [sys.executable, "-c", cmd_payload],
        log_path,
        timeout_secs=5,
        out_path=out_path,
        retry_attempts=2,
        retry_backoff_secs=0.0,
        retryable_exit_codes={126},
    )

    assert marker.exists(), "marker was not created by first attempt"
    assert "ok-retry" in output, "second attempt output missing"
    assert "ok-retry" in out_path.read_text(encoding="utf-8"), "stdout capture missing"

    # Also validate the shared env-driven retry wrapper to prevent drift
    # between script wrappers and common retry policy parsing.
    marker_env = tmp_dir / "retry-env.marker"
    log_path_env = tmp_dir / "runner-env.log"
    out_path_env = tmp_dir / "runner-env.out"
    cmd_payload_env = "\n".join(
        [
            "import pathlib",
            "import sys",
            f"marker = pathlib.Path({str(marker_env)!r})",
            "if not marker.exists():",
            "    marker.write_text('1', encoding='utf-8')",
            "    print('resource temporarily unavailable', file=sys.stderr)",
            "    raise SystemExit(126)",
            "print('ok-env-retry')",
        ]
    )
    env = {
        "FORMAL_LAUNCH_RETRY_ATTEMPTS": "2",
        "FORMAL_LAUNCH_RETRY_BACKOFF_SECS": "0",
        "FORMAL_LAUNCH_RETRYABLE_EXIT_CODES": "126",
        "FORMAL_LAUNCH_RETRYABLE_PATTERNS": "resource temporarily unavailable",
    }
    output_env = runner_common.run_command_logged_with_env_retry(
        [sys.executable, "-c", cmd_payload_env],
        log_path_env,
        timeout_secs=5,
        out_path=out_path_env,
        env=env,
    )
    assert marker_env.exists(), "env marker missing"
    assert "ok-env-retry" in output_env, "env retry output missing"
    assert "ok-env-retry" in out_path_env.read_text(
        encoding="utf-8"
    ), "env stdout capture missing"

    # Validate centralized log truncation support used by multiple runners.
    cap_log_path = tmp_dir / "runner-cap.log"
    noisy_payload = "X" * 512
    runner_common.run_command_logged(
        [sys.executable, "-c", f"print({noisy_payload!r})"],
        cap_log_path,
        timeout_secs=5,
        max_log_bytes=128,
        truncation_label="formal_runner_common_retry",
    )
    cap_log_size = cap_log_path.stat().st_size
    assert cap_log_size <= 128, f"log cap exceeded: {cap_log_size}"
    cap_log_text = cap_log_path.read_text(encoding="utf-8")
    assert (
        "log truncated from" in cap_log_text
    ), "missing truncation marker in capped log"

    # Ensure env-driven retry wrapper forwards log-cap controls.
    cap_env_log_path = tmp_dir / "runner-env-cap.log"
    runner_common.run_command_logged_with_env_retry(
        [sys.executable, "-c", f"print({noisy_payload!r})"],
        cap_env_log_path,
        timeout_secs=5,
        env={"FORMAL_LAUNCH_RETRY_ATTEMPTS": "1"},
        max_log_bytes=96,
        truncation_label="formal_runner_common_retry_env",
    )
    cap_env_log_size = cap_env_log_path.stat().st_size
    assert cap_env_log_size <= 96, f"env log cap exceeded: {cap_env_log_size}"
    cap_env_log_text = cap_env_log_path.read_text(encoding="utf-8")
    assert (
        "log truncated from" in cap_env_log_text
    ), "missing env truncation marker in capped log"

    print("PASS: shared formal runner retry helper")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    print("PASS: shared formal runner retry helper")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Exercise shared formal runner drop-reason helpers."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: formal_runner_common_drop_reasons.py <helper_dir>", file=sys.stderr)
        return 2

    helper_dir = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(helper_dir))
    import runner_common

    sample = "\n".join(
        [
            "foo.sv:12:9: warning: this construct will be dropped during lowering; id=42",
            "foo.sv:13:9: warning: this construct will be dropped during lowering; id=84",
            "bar.sv:2: warning: unrelated",
        ]
    )
    reasons = runner_common.extract_drop_reasons(
        sample, "will be dropped during lowering"
    )
    assert len(reasons) == 1, f"expected deduplicated reason, got {reasons!r}"
    assert reasons[0] == "this construct will be dropped during lowering, id=<n>"
    print("PASS: shared formal runner drop-reason helpers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

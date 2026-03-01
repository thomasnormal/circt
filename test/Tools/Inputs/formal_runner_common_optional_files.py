#!/usr/bin/env python3
"""Exercise shared formal runner optional file/allowlist helpers."""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: formal_runner_common_optional_files.py <helper_dir> <tmp_dir>",
            file=sys.stderr,
        )
        return 2

    helper_dir = Path(sys.argv[1]).resolve()
    tmp_dir = Path(sys.argv[2]).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(helper_dir))
    import runner_common

    # Empty optional paths should be accepted and map to no data.
    no_file = runner_common.resolve_optional_existing_file(
        "",
        missing_file_prefix="unused missing prefix",
    )
    assert no_file is None, "empty optional path should return None"
    no_allow_path, no_allow = runner_common.load_optional_allowlist(
        "",
        missing_file_prefix="unused missing prefix",
    )
    assert no_allow_path is None, "empty optional allowlist path should return None path"
    assert no_allow == (set(), [], []), "empty optional allowlist should be empty"
    no_empty_file = runner_common.write_optional_empty_file("")
    assert no_empty_file is None, "empty optional output path should return None"

    allow_path = tmp_dir / "allow.tsv"
    allow_path.write_text(
        "\n".join(
            [
                "exact:rule_exact",
                "prefix:group::",
                r"regex:^rule_[0-9]+$",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = runner_common.resolve_optional_existing_file(
        str(allow_path),
        missing_file_prefix="allowlist file not found",
    )
    assert resolved == allow_path.resolve(), "resolved optional file path mismatch"

    loaded_path, allowlist = runner_common.load_optional_allowlist(
        str(allow_path),
        missing_file_prefix="allowlist file not found",
    )
    assert loaded_path == allow_path.resolve(), "loaded allowlist path mismatch"
    assert runner_common.is_allowlisted("rule_exact", allowlist), "exact allowlist miss"
    assert runner_common.is_allowlisted(
        "group::member", allowlist
    ), "prefix allowlist miss"
    assert runner_common.is_allowlisted("rule_42", allowlist), "regex allowlist miss"
    assert not runner_common.is_allowlisted(
        "other_token", allowlist
    ), "unexpected allowlist hit"

    empty_out = tmp_dir / "nested" / "empty.jsonl"
    written_empty_out = runner_common.write_optional_empty_file(str(empty_out))
    assert (
        written_empty_out == empty_out.resolve()
    ), "optional empty output path mismatch"
    assert empty_out.is_file(), "optional empty output file missing"
    assert empty_out.read_text(encoding="utf-8") == "", "optional empty output not empty"

    missing_prefix = "missing optional helper file"
    missing_path = tmp_dir / "missing.tsv"
    missing_stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(missing_stderr):
            runner_common.resolve_optional_existing_file(
                str(missing_path),
                missing_file_prefix=missing_prefix,
            )
        raise AssertionError("expected missing file failure")
    except SystemExit as exc:
        assert exc.code == 1, f"unexpected missing file exit code: {exc.code}"
        msg = missing_stderr.getvalue()
        assert missing_prefix in msg, f"missing prefix not present: {msg}"
        assert str(missing_path.resolve()) in msg, f"missing path not present: {msg}"

    bad_regex_path = tmp_dir / "bad_allow.tsv"
    bad_regex_path.write_text("regex:([\n", encoding="utf-8")
    bad_regex_stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(bad_regex_stderr):
            runner_common.load_optional_allowlist(
                str(bad_regex_path),
                missing_file_prefix="allowlist file not found",
            )
        raise AssertionError("expected invalid regex failure")
    except SystemExit as exc:
        assert exc.code == 1, f"unexpected bad regex exit code: {exc.code}"
        msg = bad_regex_stderr.getvalue()
        assert "invalid allowlist row 1: bad regex" in msg, msg

    print("PASS: shared formal runner optional file helpers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

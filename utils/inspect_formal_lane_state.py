#!/usr/bin/env python3
"""Inspect and validate formal lane-state TSV artifacts."""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

TIMESTAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
CONFIG_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class LaneRow:
  lane_id: str
  suite: str
  mode: str
  total: int
  passed: int
  failed: int
  xfailed: int
  xpassed: int
  errored: int
  skipped: int
  updated_at_utc: str
  compat_policy_version: str
  config_hash: str
  source: str
  line_no: int


@dataclass(frozen=True)
class SuiteModeAggregate:
  suite: str
  mode: str
  lanes: int
  total: int
  passed: int
  failed: int
  xfailed: int
  xpassed: int
  errored: int
  skipped: int


def fail(message: str) -> None:
  raise SystemExit(f"error: {message}")


def parse_non_negative_int(value: str, field: str) -> int:
  if not re.fullmatch(r"[0-9]+", value):
    fail(f"{field} must be non-negative integer")
  return int(value)


def parse_config_hash(value: str, field: str) -> str:
  if not value:
    return ""
  if not CONFIG_HASH_RE.fullmatch(value):
    fail(f"{field} must be 64 lowercase hex chars")
  return value


def parse_lane_row(parts: list[str], source: str, line_no: int) -> LaneRow:
  if len(parts) == 11:
    parts = parts + ["", ""]
  elif len(parts) == 12:
    # Legacy format with no explicit compat_policy_version.
    parts = parts[:-1] + ["", parts[-1]]
  elif len(parts) != 13:
    fail(f"{source}:{line_no}: expected 11, 12, or 13 tab-separated fields")

  (
      lane_id,
      suite,
      mode,
      total,
      passed,
      failed,
      xfailed,
      xpassed,
      errored,
      skipped,
      updated_at_utc,
      compat_policy_version,
      config_hash,
  ) = parts

  if not lane_id:
    fail(f"{source}:{line_no}: lane_id is required")
  if not suite or not mode:
    fail(f"{source}:{line_no}: suite and mode are required")
  if not updated_at_utc:
    fail(f"{source}:{line_no}: updated_at_utc is required")
  if not TIMESTAMP_RE.fullmatch(updated_at_utc):
    fail(f"{source}:{line_no}: updated_at_utc must be UTC RFC3339 timestamp")
  if (
      compat_policy_version
      and compat_policy_version != "legacy"
      and not re.fullmatch(r"[0-9]+", compat_policy_version)
  ):
    fail(f"{source}:{line_no}: compat_policy_version must be integer")

  return LaneRow(
      lane_id=lane_id,
      suite=suite,
      mode=mode,
      total=parse_non_negative_int(total, f"{source}:{line_no}: total"),
      passed=parse_non_negative_int(passed, f"{source}:{line_no}: pass"),
      failed=parse_non_negative_int(failed, f"{source}:{line_no}: fail"),
      xfailed=parse_non_negative_int(xfailed, f"{source}:{line_no}: xfail"),
      xpassed=parse_non_negative_int(xpassed, f"{source}:{line_no}: xpass"),
      errored=parse_non_negative_int(errored, f"{source}:{line_no}: error"),
      skipped=parse_non_negative_int(skipped, f"{source}:{line_no}: skip"),
      updated_at_utc=updated_at_utc,
      compat_policy_version=compat_policy_version,
      config_hash=parse_config_hash(config_hash, f"{source}:{line_no}: config_hash"),
      source=source,
      line_no=line_no,
  )


def row_payload_tuple(row: LaneRow) -> tuple[str, ...]:
  return (
      row.lane_id,
      row.suite,
      row.mode,
      str(row.total),
      str(row.passed),
      str(row.failed),
      str(row.xfailed),
      str(row.xpassed),
      str(row.errored),
      str(row.skipped),
      row.updated_at_utc,
      row.compat_policy_version,
      row.config_hash,
  )


def load_rows(path: Path) -> list[LaneRow]:
  if not path.is_file():
    fail(f"lane-state file not found: {path}")

  rows: list[LaneRow] = []
  for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    if not raw_line.strip():
      continue
    parts = raw_line.split("\t")
    if line_no == 1 and parts and parts[0] == "lane_id":
      continue
    rows.append(parse_lane_row(parts, str(path), line_no))
  return rows


def merge_lane_rows(rows: list[LaneRow]) -> tuple[dict[str, LaneRow], int]:
  merged: dict[str, LaneRow] = {}
  duplicate_lane_rows = 0

  for row in rows:
    current = merged.get(row.lane_id)
    if current is None:
      merged[row.lane_id] = row
      continue

    duplicate_lane_rows += 1

    if (
        current.compat_policy_version
        and row.compat_policy_version
        and current.compat_policy_version != row.compat_policy_version
    ):
      fail(
          "conflicting compat_policy_version for lane "
          f"{row.lane_id}: {current.source}:{current.line_no} has "
          f"{current.compat_policy_version}, {row.source}:{row.line_no} has "
          f"{row.compat_policy_version}"
      )

    if not current.compat_policy_version and row.compat_policy_version:
      merged[row.lane_id] = row
      continue
    if current.compat_policy_version and not row.compat_policy_version:
      continue

    if current.config_hash and row.config_hash and current.config_hash != row.config_hash:
      fail(
          "conflicting config_hash for lane "
          f"{row.lane_id}: {current.source}:{current.line_no} has {current.config_hash}, "
          f"{row.source}:{row.line_no} has {row.config_hash}"
      )

    if not current.config_hash and row.config_hash:
      merged[row.lane_id] = row
      continue
    if current.config_hash and not row.config_hash:
      continue

    if row.updated_at_utc > current.updated_at_utc:
      merged[row.lane_id] = row
      continue
    if row.updated_at_utc < current.updated_at_utc:
      continue

    if row_payload_tuple(current) != row_payload_tuple(row):
      fail(
          "conflicting row payload for lane "
          f"{row.lane_id}: equal updated_at_utc={row.updated_at_utc} but "
          f"different payloads at {current.source}:{current.line_no} and "
          f"{row.source}:{row.line_no}"
      )

  return merged, duplicate_lane_rows


def build_suite_mode_aggregates(rows: list[LaneRow]) -> list[SuiteModeAggregate]:
  grouped: dict[tuple[str, str], list[LaneRow]] = {}
  for row in rows:
    grouped.setdefault((row.suite, row.mode), []).append(row)

  result: list[SuiteModeAggregate] = []
  for suite, mode in sorted(grouped):
    group_rows = grouped[(suite, mode)]
    result.append(
        SuiteModeAggregate(
            suite=suite,
            mode=mode,
            lanes=len(group_rows),
            total=sum(r.total for r in group_rows),
            passed=sum(r.passed for r in group_rows),
            failed=sum(r.failed for r in group_rows),
            xfailed=sum(r.xfailed for r in group_rows),
            xpassed=sum(r.xpassed for r in group_rows),
            errored=sum(r.errored for r in group_rows),
            skipped=sum(r.skipped for r in group_rows),
        )
    )
  return result


def build_json_payload(
    files: list[Path],
    input_rows: int,
    duplicate_lane_rows: int,
    merged_rows: list[LaneRow],
    observed_compat_policy_versions: list[str],
    observed_config_hashes: list[str],
    suite_mode_aggregates: list[SuiteModeAggregate],
) -> dict:
  return {
      "schema_version": 1,
      "input_files": [str(path) for path in files],
      "input_rows": input_rows,
      "merged_lanes": len(merged_rows),
      "duplicate_lane_rows": duplicate_lane_rows,
      "observed_compat_policy_versions": observed_compat_policy_versions,
      "observed_config_hashes": observed_config_hashes,
      "counters": {
          "total": sum(r.total for r in merged_rows),
          "pass": sum(r.passed for r in merged_rows),
          "fail": sum(r.failed for r in merged_rows),
          "xfail": sum(r.xfailed for r in merged_rows),
          "xpass": sum(r.xpassed for r in merged_rows),
          "error": sum(r.errored for r in merged_rows),
          "skip": sum(r.skipped for r in merged_rows),
      },
      "suite_mode": [
          {
              "suite": row.suite,
              "mode": row.mode,
              "lanes": row.lanes,
              "total": row.total,
              "pass": row.passed,
              "fail": row.failed,
              "xfail": row.xfailed,
              "xpass": row.xpassed,
              "error": row.errored,
              "skip": row.skipped,
          }
          for row in suite_mode_aggregates
      ],
      "lanes": [
          {
              "lane_id": row.lane_id,
              "suite": row.suite,
              "mode": row.mode,
              "total": row.total,
              "pass": row.passed,
              "fail": row.failed,
              "xfail": row.xfailed,
              "xpass": row.xpassed,
              "error": row.errored,
              "skip": row.skipped,
              "updated_at_utc": row.updated_at_utc,
              "compat_policy_version": row.compat_policy_version,
              "config_hash": row.config_hash,
              "source": row.source,
              "line_no": row.line_no,
          }
          for row in merged_rows
      ],
  }


def main() -> int:
  parser = argparse.ArgumentParser(
      description="Inspect and validate formal lane-state TSV artifacts."
  )
  parser.add_argument("lane_state_tsv", nargs="+", help="Input lane-state TSV files")
  parser.add_argument("--json-out", help="Write JSON summary to file")
  parser.add_argument(
      "--print-lanes",
      action="store_true",
      help="Print merged lane rows in stable order",
  )
  parser.add_argument(
      "--require-config-hash",
      action="store_true",
      help="Fail if any merged lane row has empty config_hash",
  )
  parser.add_argument(
      "--require-compat-policy-version",
      action="store_true",
      help="Fail if any merged lane row has empty compat_policy_version",
  )
  parser.add_argument(
      "--expect-compat-policy-version",
      help="Fail if any non-empty compat_policy_version differs from this value",
  )
  parser.add_argument(
      "--require-single-config-hash",
      action="store_true",
      help="Fail if non-empty observed config_hash set has more than one value",
  )
  parser.add_argument(
      "--require-lane",
      action="append",
      default=[],
      help="Require merged lane_id to exist (repeatable)",
  )
  args = parser.parse_args()

  files = [Path(path) for path in args.lane_state_tsv]
  all_rows: list[LaneRow] = []
  for path in files:
    all_rows.extend(load_rows(path))

  merged, duplicate_lane_rows = merge_lane_rows(all_rows)
  merged_rows = [merged[lane_id] for lane_id in sorted(merged)]
  observed_compat_policy_versions = sorted(
      {row.compat_policy_version for row in all_rows if row.compat_policy_version}
  )
  observed_hashes = sorted({row.config_hash for row in all_rows if row.config_hash})

  if args.require_single_config_hash and len(observed_hashes) > 1:
    fail(
        "multiple non-empty config_hash values observed: "
        + ",".join(observed_hashes)
    )

  if args.require_config_hash:
    missing = [row.lane_id for row in merged_rows if not row.config_hash]
    if missing:
      fail("missing config_hash for lane(s): " + ", ".join(missing))

  if args.require_compat_policy_version:
    missing = [
        row.lane_id
        for row in merged_rows
        if not row.compat_policy_version or row.compat_policy_version == "legacy"
    ]
    if missing:
      fail("missing compat_policy_version for lane(s): " + ", ".join(missing))

  if args.expect_compat_policy_version is not None:
    expected = args.expect_compat_policy_version.strip()
    if not expected:
      fail("--expect-compat-policy-version must be non-empty")
    mismatched = [
        row.lane_id
        for row in merged_rows
        if row.compat_policy_version and row.compat_policy_version != expected
    ]
    if mismatched:
      fail("unexpected compat_policy_version for lane(s): " + ", ".join(mismatched))

  if args.require_lane:
    missing = [lane_id for lane_id in args.require_lane if lane_id not in merged]
    if missing:
      fail("required lane(s) missing: " + ", ".join(missing))

  suite_mode_aggregates = build_suite_mode_aggregates(merged_rows)

  summary_total = sum(r.total for r in merged_rows)
  summary_pass = sum(r.passed for r in merged_rows)
  summary_fail = sum(r.failed for r in merged_rows)
  summary_xfail = sum(r.xfailed for r in merged_rows)
  summary_xpass = sum(r.xpassed for r in merged_rows)
  summary_error = sum(r.errored for r in merged_rows)
  summary_skip = sum(r.skipped for r in merged_rows)

  print(
      "SUMMARY "
      f"input_files={len(files)} "
      f"input_rows={len(all_rows)} "
      f"merged_lanes={len(merged_rows)} "
      f"duplicate_lane_rows={duplicate_lane_rows}"
  )
  print(
      "SUMMARY "
      f"compat_policy_versions_non_empty={len(observed_compat_policy_versions)} "
      "compat_policy_version_list="
      + (",".join(observed_compat_policy_versions)
         if observed_compat_policy_versions
         else "-")
  )
  print(
      "SUMMARY "
      f"config_hashes_non_empty={len(observed_hashes)} "
      "config_hash_list="
      + (",".join(observed_hashes) if observed_hashes else "-")
  )
  print(
      "SUMMARY counters "
      f"total={summary_total} "
      f"pass={summary_pass} "
      f"fail={summary_fail} "
      f"xfail={summary_xfail} "
      f"xpass={summary_xpass} "
      f"error={summary_error} "
      f"skip={summary_skip}"
  )

  for row in suite_mode_aggregates:
    print(
        "SUITE_MODE "
        f"suite={row.suite} "
        f"mode={row.mode} "
        f"lanes={row.lanes} "
        f"total={row.total} "
        f"pass={row.passed} "
        f"fail={row.failed} "
        f"xfail={row.xfailed} "
        f"xpass={row.xpassed} "
        f"error={row.errored} "
        f"skip={row.skipped}"
    )

  if args.print_lanes:
    for row in merged_rows:
      print(
          "LANE "
          f"lane_id={row.lane_id} "
          f"suite={row.suite} "
          f"mode={row.mode} "
          f"total={row.total} "
          f"pass={row.passed} "
          f"fail={row.failed} "
          f"xfail={row.xfailed} "
          f"xpass={row.xpassed} "
          f"error={row.errored} "
          f"skip={row.skipped} "
          f"updated_at_utc={row.updated_at_utc} "
          f"compat_policy_version={row.compat_policy_version or '-'} "
          f"config_hash={row.config_hash or '-'}"
      )

  if args.json_out:
    payload = build_json_payload(
        files,
        len(all_rows),
        duplicate_lane_rows,
        merged_rows,
        observed_compat_policy_versions,
        observed_hashes,
        suite_mode_aggregates,
    )
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

  return 0


if __name__ == "__main__":
  sys.exit(main())

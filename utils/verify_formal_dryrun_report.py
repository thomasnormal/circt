#!/usr/bin/env python3
"""Verify run-level integrity in formal dry-run JSONL reports."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def fail(message: str) -> None:
  raise SystemExit(f"error: {message}")


def canonical_row(row: dict) -> bytes:
  payload = {k: v for k, v in row.items() if not k.startswith("_")}
  return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def hash_rows(rows: list[dict]) -> str:
  digest = hashlib.sha256()
  for row in rows:
    digest.update(canonical_row(row))
    digest.update(b"\n")
  return digest.hexdigest()


def verify_report(path: Path) -> int:
  rows: list[dict] = []
  for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
      continue
    try:
      row = json.loads(line)
    except Exception as ex:
      fail(f"{path}:{line_no}: invalid JSON ({ex})")
    if not isinstance(row, dict):
      fail(f"{path}:{line_no}: expected JSON object row")
    row["_line_no"] = line_no
    rows.append(row)

  if not rows:
    fail(f"{path}: report is empty")

  run_count = 0
  idx = 0
  while idx < len(rows):
    row = rows[idx]
    line_no = row["_line_no"]
    if row.get("operation") != "run_meta":
      fail(f"{path}:{line_no}: expected operation=run_meta")
    run_id = row.get("run_id")
    if not isinstance(run_id, str) or not run_id:
      fail(f"{path}:{line_no}: run_meta must include non-empty run_id")

    segment = [row]
    idx += 1
    found_end = False
    while idx < len(rows):
      cur = rows[idx]
      cur_line = cur["_line_no"]
      if cur.get("operation") == "run_meta":
        break
      if cur.get("run_id") != run_id:
        fail(
            f"{path}:{cur_line}: run_id mismatch; expected '{run_id}', got "
            f"'{cur.get('run_id', '')}'"
        )
      segment.append(cur)
      idx += 1
      if cur.get("operation") == "run_end":
        found_end = True
        break

    if not found_end:
      fail(f"{path}:{line_no}: missing run_end for run_id '{run_id}'")

    run_end = segment[-1]
    pre_end_rows = segment[:-1]
    end_line = run_end["_line_no"]
    expected_count = run_end.get("row_count")
    if not isinstance(expected_count, int):
      fail(f"{path}:{end_line}: run_end.row_count must be integer")
    if expected_count != len(pre_end_rows):
      fail(
          f"{path}:{end_line}: run_end.row_count={expected_count} does not match "
          f"observed rows={len(pre_end_rows)}"
      )

    expected_hash = run_end.get("payload_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
      fail(f"{path}:{end_line}: run_end.payload_sha256 must be non-empty string")
    actual_hash = hash_rows(pre_end_rows)
    if expected_hash != actual_hash:
      fail(
          f"{path}:{end_line}: run_end.payload_sha256 mismatch; expected "
          f"{actual_hash}, got {expected_hash}"
      )

    exit_code = run_end.get("exit_code")
    if not isinstance(exit_code, int):
      fail(f"{path}:{end_line}: run_end.exit_code must be integer")

    run_count += 1

  return run_count


def main() -> int:
  parser = argparse.ArgumentParser(
      description="Verify integrity for formal dry-run JSONL reports."
  )
  parser.add_argument("report_jsonl", help="Path to report JSONL file")
  args = parser.parse_args()

  report_path = Path(args.report_jsonl)
  if not report_path.is_file():
    fail(f"report not found: {report_path}")

  run_count = verify_report(report_path)
  print(f"verified dry-run report: runs={run_count}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

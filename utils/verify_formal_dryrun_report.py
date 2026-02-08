#!/usr/bin/env python3
"""Verify run-level integrity in formal dry-run JSONL reports."""

import argparse
import hmac
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional


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


def read_hmac_keyring(path: Path) -> dict[str, bytes]:
  if not path.is_file():
    fail(f"HMAC keyring file not found: {path}")
  keyring: dict[str, bytes] = {}
  for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    fields = line.split("\t")
    if len(fields) != 2:
      fail(
          f"{path}:{line_no}: expected '<hmac_key_id>\\t<key_file_path>' row in keyring"
      )
    key_id = fields[0].strip()
    key_file = fields[1].strip()
    if not key_id:
      fail(f"{path}:{line_no}: empty hmac_key_id in keyring row")
    if key_id in keyring:
      fail(f"{path}:{line_no}: duplicate hmac_key_id '{key_id}' in keyring")
    key_path = Path(key_file)
    if not key_path.is_file():
      fail(f"{path}:{line_no}: HMAC key file not found: {key_path}")
    keyring[key_id] = key_path.read_bytes()
  if not keyring:
    fail(f"{path}: keyring has no usable rows")
  return keyring


def verify_report(
    path: Path,
    allow_legacy_prefix: bool,
    hmac_key_bytes: Optional[bytes],
    hmac_keyring: Optional[dict[str, bytes]],
    expected_hmac_key_id: Optional[str],
) -> int:
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

  if allow_legacy_prefix:
    first_run_meta = None
    for idx, row in enumerate(rows):
      if row.get("operation") == "run_meta":
        first_run_meta = idx
        break
    if first_run_meta is None:
      fail(f"{path}: no run_meta rows found")
    rows = rows[first_run_meta:]

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
    run_meta_hmac_key_id = row.get("hmac_key_id")
    if run_meta_hmac_key_id is None:
      run_meta_hmac_key_id = ""
    if not isinstance(run_meta_hmac_key_id, str):
      fail(f"{path}:{line_no}: run_meta.hmac_key_id must be string")
    if expected_hmac_key_id is not None and run_meta_hmac_key_id != expected_hmac_key_id:
      fail(
          f"{path}:{line_no}: run_meta.hmac_key_id mismatch; expected "
          f"'{expected_hmac_key_id}', got '{run_meta_hmac_key_id}'"
      )
    run_hmac_key_bytes = hmac_key_bytes
    if hmac_keyring is not None:
      if not run_meta_hmac_key_id:
        fail(
            f"{path}:{line_no}: run_meta.hmac_key_id must be non-empty when using "
            "--hmac-keyring-tsv"
        )
      run_hmac_key_bytes = hmac_keyring.get(run_meta_hmac_key_id)
      if run_hmac_key_bytes is None:
        fail(
            f"{path}:{line_no}: unknown hmac_key_id '{run_meta_hmac_key_id}' "
            f"(not found in keyring)"
        )

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

    if run_hmac_key_bytes is not None:
      actual_hmac = run_end.get("payload_hmac_sha256")
      if not isinstance(actual_hmac, str) or not actual_hmac:
        fail(f"{path}:{end_line}: run_end.payload_hmac_sha256 must be non-empty string")
      expected_hmac = hmac.new(
          run_hmac_key_bytes, actual_hash.encode("utf-8"), hashlib.sha256
      ).hexdigest()
      if actual_hmac != expected_hmac:
        fail(
            f"{path}:{end_line}: run_end.payload_hmac_sha256 mismatch; expected "
            f"{expected_hmac}, got {actual_hmac}"
        )
    run_end_hmac_key_id = run_end.get("hmac_key_id")
    if run_end_hmac_key_id is None:
      run_end_hmac_key_id = ""
    if not isinstance(run_end_hmac_key_id, str):
      fail(f"{path}:{end_line}: run_end.hmac_key_id must be string")
    if run_end_hmac_key_id != run_meta_hmac_key_id:
      fail(
          f"{path}:{end_line}: run_end.hmac_key_id mismatch; expected "
          f"'{run_meta_hmac_key_id}', got '{run_end_hmac_key_id}'"
      )
    if expected_hmac_key_id is not None and run_end_hmac_key_id != expected_hmac_key_id:
      fail(
          f"{path}:{end_line}: run_end.hmac_key_id mismatch; expected "
          f"'{expected_hmac_key_id}', got '{run_end_hmac_key_id}'"
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
  parser.add_argument(
      "--allow-legacy-prefix",
      action="store_true",
      help="Ignore leading pre-run_meta legacy rows before validating segments",
  )
  parser.add_argument(
      "--hmac-key-file",
      help="Optional key file for validating run_end.payload_hmac_sha256",
  )
  parser.add_argument(
      "--hmac-keyring-tsv",
      help="Optional TSV mapping '<hmac_key_id>\\t<key_file_path>' for keyed HMAC validation",
  )
  parser.add_argument(
      "--expected-hmac-key-id",
      help="Optional expected hmac_key_id for run_meta/run_end rows",
  )
  args = parser.parse_args()

  report_path = Path(args.report_jsonl)
  if not report_path.is_file():
    fail(f"report not found: {report_path}")

  hmac_key_bytes = None
  if args.hmac_key_file:
    key_path = Path(args.hmac_key_file)
    if not key_path.is_file():
      fail(f"HMAC key file not found: {key_path}")
    hmac_key_bytes = key_path.read_bytes()
  hmac_keyring = None
  if args.hmac_keyring_tsv:
    hmac_keyring = read_hmac_keyring(Path(args.hmac_keyring_tsv))
  if hmac_key_bytes is not None and hmac_keyring is not None:
    fail("cannot use both --hmac-key-file and --hmac-keyring-tsv")

  run_count = verify_report(
      report_path,
      allow_legacy_prefix=args.allow_legacy_prefix,
      hmac_key_bytes=hmac_key_bytes,
      hmac_keyring=hmac_keyring,
      expected_hmac_key_id=args.expected_hmac_key_id,
  )
  print(f"verified dry-run report: runs={run_count}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

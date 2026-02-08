#!/usr/bin/env python3
"""Verify run-level integrity in formal dry-run JSONL reports."""

import argparse
import base64
import binascii
import hmac
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import date
from string import hexdigits
from pathlib import Path
from typing import Optional


def fail(message: str) -> None:
  raise SystemExit(f"error: {message}")


def canonical_json_bytes(payload: dict) -> bytes:
  return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_row(row: dict) -> bytes:
  payload = {k: v for k, v in row.items() if not k.startswith("_")}
  return canonical_json_bytes(payload)


def hash_rows(rows: list[dict]) -> str:
  digest = hashlib.sha256()
  for row in rows:
    digest.update(canonical_row(row))
    digest.update(b"\n")
  return digest.hexdigest()


def parse_iso_date(value: object, field_name: str) -> Optional[date]:
  if value is None:
    return None
  if not isinstance(value, str):
    fail(f"{field_name} must be YYYY-MM-DD string")
  raw = value.strip()
  if not raw:
    return None
  try:
    return date.fromisoformat(raw)
  except ValueError:
    fail(f"{field_name} must be YYYY-MM-DD string")


def parse_sha256_hex(value: object, field_name: str) -> str:
  if not isinstance(value, str):
    fail(f"{field_name} must be 64-character hex SHA-256")
  raw = value.strip().lower()
  if len(raw) != 64 or any(ch not in hexdigits for ch in raw):
    fail(f"{field_name} must be 64-character hex SHA-256")
  return raw


def parse_non_empty_string(value: object, field_name: str) -> str:
  if not isinstance(value, str):
    fail(f"{field_name} must be non-empty string")
  raw = value.strip()
  if not raw:
    fail(f"{field_name} must be non-empty string")
  return raw


def parse_base64_bytes(value: object, field_name: str) -> bytes:
  if not isinstance(value, str):
    fail(f"{field_name} must be non-empty base64 string")
  raw = value.strip()
  if not raw:
    fail(f"{field_name} must be non-empty base64 string")
  try:
    return base64.b64decode(raw, validate=True)
  except (binascii.Error, ValueError):
    fail(f"{field_name} must be non-empty base64 string")


def parse_schema_version_one(value: object, field_name: str) -> None:
  if value is None:
    return
  if value != 1:
    fail(f"{field_name} must be 1")


def verify_ed25519_manifest_signature(
    manifest_path: Path,
    manifest_payload: dict,
    signature_base64: object,
    public_key_path: Path,
) -> None:
  if not public_key_path.is_file():
    fail(f"manifest Ed25519 public key file not found: {public_key_path}")
  signature_bytes = parse_base64_bytes(
      signature_base64,
      f"{manifest_path}: manifest.signature_ed25519_base64",
  )
  payload_bytes = canonical_json_bytes(manifest_payload)
  with tempfile.TemporaryDirectory(prefix="formal-manifest-verify-") as temp_dir:
    temp_dir_path = Path(temp_dir)
    payload_file = temp_dir_path / "manifest_payload.json"
    signature_file = temp_dir_path / "manifest_signature.bin"
    payload_file.write_bytes(payload_bytes)
    signature_file.write_bytes(signature_bytes)
    verify_proc = subprocess.run(
        [
            "openssl",
            "pkeyutl",
            "-verify",
            "-pubin",
            "-inkey",
            str(public_key_path),
            "-sigfile",
            str(signature_file),
            "-rawin",
            "-in",
            str(payload_file),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
  if verify_proc.returncode != 0:
    fail(
        f"{manifest_path}: manifest.signature_ed25519_base64 verification failed "
        f"for {public_key_path}"
    )


def read_ed25519_signer_keyring(
    path: Path,
) -> dict[str, tuple[Path, Optional[date], Optional[date], str]]:
  if not path.is_file():
    fail(f"manifest Ed25519 signer keyring file not found: {path}")
  keyring: dict[str, tuple[Path, Optional[date], Optional[date], str]] = {}
  for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    fields = line.split("\t")
    if len(fields) < 2 or len(fields) > 6:
      fail(
          f"{path}:{line_no}: expected '<signer_id>\\t<public_key_file_path>"
          "\\t[not_before]\\t[not_after]\\t[status]\\t[key_sha256]' row in "
          "manifest Ed25519 signer keyring"
      )
    signer_id = fields[0].strip()
    key_file = fields[1].strip()
    if not signer_id:
      fail(f"{path}:{line_no}: empty signer_id in manifest Ed25519 signer keyring row")
    if signer_id in keyring:
      fail(
          f"{path}:{line_no}: duplicate signer_id '{signer_id}' in "
          "manifest Ed25519 signer keyring"
      )
    key_path = Path(key_file)
    if not key_path.is_absolute():
      key_path = path.parent / key_path
    if not key_path.is_file():
      fail(f"{path}:{line_no}: manifest Ed25519 public key file not found: {key_path}")
    key_bytes = key_path.read_bytes()
    not_before = None
    not_after = None
    if len(fields) >= 3:
      not_before = parse_iso_date(
          fields[2], f"{path}:{line_no}: signer_keyring.not_before"
      )
    if len(fields) >= 4:
      not_after = parse_iso_date(
          fields[3], f"{path}:{line_no}: signer_keyring.not_after"
      )
    if not_before is not None and not_after is not None and not_before > not_after:
      fail(f"{path}:{line_no}: signer_keyring.not_before must be <= signer_keyring.not_after")
    status = "active"
    if len(fields) >= 5:
      status = fields[4].strip().lower()
      if not status:
        status = "active"
      if status not in {"active", "revoked"}:
        fail(
            f"{path}:{line_no}: signer_keyring.status must be one of active, revoked"
        )
    if len(fields) >= 6:
      expected_key_hash = parse_sha256_hex(
          fields[5], f"{path}:{line_no}: signer_keyring.key_sha256"
      )
      actual_key_hash = hashlib.sha256(key_bytes).hexdigest()
      if actual_key_hash != expected_key_hash:
        fail(
            f"{path}:{line_no}: signer_keyring.key_sha256 mismatch; expected "
            f"{expected_key_hash}, got {actual_key_hash}"
        )
    keyring[signer_id] = (key_path, not_before, not_after, status)
  if not keyring:
    fail(f"{path}: manifest Ed25519 signer keyring has no usable rows")
  return keyring


def read_keyring_manifest(
    manifest_path: Path,
    manifest_hmac_key_bytes: Optional[bytes],
    manifest_ed25519_public_key_path: Optional[Path],
    manifest_ed25519_signer_keyring: Optional[dict[str, tuple[Path, Optional[date], Optional[date], str]]],
    expected_signer_id: Optional[str],
) -> str:
  mode_count = int(manifest_hmac_key_bytes is not None)
  mode_count += int(manifest_ed25519_public_key_path is not None)
  mode_count += int(manifest_ed25519_signer_keyring is not None)
  if mode_count != 1:
    fail(
        "internal: exactly one of manifest_hmac_key_bytes, "
        "manifest_ed25519_public_key_path, or "
        "manifest_ed25519_signer_keyring must be set"
    )
  if not manifest_path.is_file():
    fail(f"keyring manifest file not found: {manifest_path}")
  try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  except Exception as ex:
    fail(f"{manifest_path}: invalid JSON ({ex})")
  if not isinstance(manifest, dict):
    fail(f"{manifest_path}: expected JSON object")

  parse_schema_version_one(
      manifest.get("schema_version"), f"{manifest_path}: manifest.schema_version"
  )
  signer_id = parse_non_empty_string(
      manifest.get("signer_id"), f"{manifest_path}: manifest.signer_id"
  )
  if expected_signer_id is not None and signer_id != expected_signer_id:
    fail(
        f"{manifest_path}: manifest.signer_id mismatch; expected "
        f"'{expected_signer_id}', got '{signer_id}'"
    )
  keyring_sha256 = parse_sha256_hex(
      manifest.get("keyring_sha256"), f"{manifest_path}: manifest.keyring_sha256"
  )
  expires_on = parse_iso_date(
      manifest.get("expires_on"), f"{manifest_path}: manifest.expires_on"
  )
  if expires_on is not None and date.today() > expires_on:
    fail(
        f"{manifest_path}: manifest.expires_on '{expires_on.isoformat()}' is expired"
    )
  payload = {
      k: v
      for k, v in manifest.items()
      if k not in {"signature_hmac_sha256", "signature_ed25519_base64"}
  }
  if manifest_hmac_key_bytes is not None:
    if manifest.get("signature_ed25519_base64") is not None:
      fail(
          f"{manifest_path}: manifest.signature_ed25519_base64 is not allowed when "
          "using --hmac-keyring-manifest-hmac-key-file"
      )
    signature = parse_sha256_hex(
        manifest.get("signature_hmac_sha256"),
        f"{manifest_path}: manifest.signature_hmac_sha256",
    )
    expected_signature = hmac.new(
        manifest_hmac_key_bytes, canonical_json_bytes(payload), hashlib.sha256
    ).hexdigest()
    if hmac.compare_digest(signature, expected_signature):
      return keyring_sha256
    fail(
        f"{manifest_path}: manifest.signature_hmac_sha256 mismatch; expected "
        f"{expected_signature}, got {signature}"
    )
  if manifest.get("signature_hmac_sha256") is not None:
    fail(
        f"{manifest_path}: manifest.signature_hmac_sha256 is not allowed when using "
        "--hmac-keyring-manifest-ed25519 verification modes"
    )
  signer_public_key_path = manifest_ed25519_public_key_path
  if manifest_ed25519_signer_keyring is not None:
    signer_entry = manifest_ed25519_signer_keyring.get(signer_id)
    if signer_entry is None:
      fail(
          f"{manifest_path}: unknown manifest signer_id '{signer_id}' "
          "(not found in Ed25519 signer keyring)"
      )
    signer_public_key_path, not_before, not_after, status = signer_entry
    if status == "revoked":
      fail(f"{manifest_path}: manifest signer_id '{signer_id}' is revoked in signer keyring")
    today = date.today()
    if not_before is not None and today < not_before:
      fail(
          f"{manifest_path}: manifest signer_id '{signer_id}' is not yet valid "
          f"(not_before={not_before.isoformat()})"
      )
    if not_after is not None and today > not_after:
      fail(
          f"{manifest_path}: manifest signer_id '{signer_id}' is expired in signer "
          f"keyring (not_after={not_after.isoformat()})"
      )
  verify_ed25519_manifest_signature(
      manifest_path,
      payload,
      manifest.get("signature_ed25519_base64"),
      signer_public_key_path,
  )
  return keyring_sha256


def read_hmac_keyring(
    path: Path,
) -> dict[str, tuple[bytes, Optional[date], Optional[date], str]]:
  if not path.is_file():
    fail(f"HMAC keyring file not found: {path}")
  keyring: dict[str, tuple[bytes, Optional[date], Optional[date], str]] = {}
  for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    fields = line.split("\t")
    if len(fields) < 2 or len(fields) > 6:
      fail(
          f"{path}:{line_no}: expected '<hmac_key_id>\\t<key_file_path>"
          "\\t[not_before]\\t[not_after]\\t[status]\\t[key_sha256]' row in keyring"
      )
    key_id = fields[0].strip()
    key_file = fields[1].strip()
    if not key_id:
      fail(f"{path}:{line_no}: empty hmac_key_id in keyring row")
    if key_id in keyring:
      fail(f"{path}:{line_no}: duplicate hmac_key_id '{key_id}' in keyring")
    key_path = Path(key_file)
    if not key_path.is_absolute():
      key_path = path.parent / key_path
    if not key_path.is_file():
      fail(f"{path}:{line_no}: HMAC key file not found: {key_path}")
    key_bytes = key_path.read_bytes()
    not_before = None
    not_after = None
    if len(fields) >= 3:
      not_before = parse_iso_date(fields[2], f"{path}:{line_no}: keyring.not_before")
    if len(fields) >= 4:
      not_after = parse_iso_date(fields[3], f"{path}:{line_no}: keyring.not_after")
    if not_before is not None and not_after is not None and not_before > not_after:
      fail(f"{path}:{line_no}: keyring.not_before must be <= keyring.not_after")
    status = "active"
    if len(fields) >= 5:
      status = fields[4].strip().lower()
      if not status:
        status = "active"
      if status not in {"active", "revoked"}:
        fail(f"{path}:{line_no}: keyring.status must be one of active, revoked")
    if len(fields) >= 6:
      expected_key_hash = parse_sha256_hex(
          fields[5], f"{path}:{line_no}: keyring.key_sha256"
      )
      actual_key_hash = hashlib.sha256(key_bytes).hexdigest()
      if actual_key_hash != expected_key_hash:
        fail(
            f"{path}:{line_no}: keyring.key_sha256 mismatch; expected "
            f"{expected_key_hash}, got {actual_key_hash}"
        )
    keyring[key_id] = (key_bytes, not_before, not_after, status)
  if not keyring:
    fail(f"{path}: keyring has no usable rows")
  return keyring


def verify_report(
    path: Path,
    allow_legacy_prefix: bool,
    hmac_key_bytes: Optional[bytes],
    hmac_keyring: Optional[dict[str, tuple[bytes, Optional[date], Optional[date], str]]],
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
      keyring_entry = hmac_keyring.get(run_meta_hmac_key_id)
      if keyring_entry is None:
        fail(
            f"{path}:{line_no}: unknown hmac_key_id '{run_meta_hmac_key_id}' "
            f"(not found in keyring)"
        )
      run_hmac_key_bytes, not_before, not_after, status = keyring_entry
      if status == "revoked":
        fail(
            f"{path}:{line_no}: hmac_key_id '{run_meta_hmac_key_id}' is revoked in keyring"
        )
      run_date = parse_iso_date(row.get("date"), f"{path}:{line_no}: run_meta.date")
      if run_date is None:
        fail(
            f"{path}:{line_no}: run_meta.date must be non-empty YYYY-MM-DD when using "
            "--hmac-keyring-tsv"
        )
      if not_before is not None and run_date < not_before:
        fail(
            f"{path}:{line_no}: run_meta.date '{run_date.isoformat()}' is before "
            f"keyring.not_before '{not_before.isoformat()}' for hmac_key_id "
            f"'{run_meta_hmac_key_id}'"
        )
      if not_after is not None and run_date > not_after:
        fail(
            f"{path}:{line_no}: run_meta.date '{run_date.isoformat()}' is after "
            f"keyring.not_after '{not_after.isoformat()}' for hmac_key_id "
            f"'{run_meta_hmac_key_id}'"
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
      help=(
          "Optional TSV mapping "
          "'<hmac_key_id>\\t<key_file_path>\\t[not_before]\\t[not_after]\\t[status]\\t[key_sha256]' "
          "for keyed HMAC validation"
      ),
  )
  parser.add_argument(
      "--hmac-keyring-sha256",
      help="Optional expected SHA-256 for keyring file content (requires --hmac-keyring-tsv)",
  )
  parser.add_argument(
      "--hmac-keyring-manifest-json",
      help="Optional keyring manifest JSON file with signed keyring_sha256 metadata",
  )
  parser.add_argument(
      "--hmac-keyring-manifest-hmac-key-file",
      help=(
          "HMAC key file used when --hmac-keyring-manifest-json carries "
          "signature_hmac_sha256"
      ),
  )
  parser.add_argument(
      "--hmac-keyring-manifest-ed25519-public-key-file",
      help=(
          "Ed25519 public key used when --hmac-keyring-manifest-json carries "
          "signature_ed25519_base64"
      ),
  )
  parser.add_argument(
      "--hmac-keyring-manifest-ed25519-keyring-tsv",
      help=(
          "Optional Ed25519 signer keyring TSV mapping "
          "'<signer_id>\\t<public_key_file_path>\\t[not_before]\\t[not_after]"
          "\\t[status]\\t[key_sha256]'"
      ),
  )
  parser.add_argument(
      "--hmac-keyring-manifest-ed25519-keyring-sha256",
      help=(
          "Optional expected SHA-256 for "
          "--hmac-keyring-manifest-ed25519-keyring-tsv content"
      ),
  )
  parser.add_argument(
      "--expected-keyring-signer-id",
      help="Optional expected signer_id value in keyring manifest",
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
    keyring_path = Path(args.hmac_keyring_tsv)
    expected_keyring_hash = None
    if args.hmac_keyring_manifest_json:
      use_manifest_hmac_key = bool(args.hmac_keyring_manifest_hmac_key_file)
      use_manifest_ed25519_public_key = bool(
          args.hmac_keyring_manifest_ed25519_public_key_file
      )
      use_manifest_ed25519_signer_keyring = bool(
          args.hmac_keyring_manifest_ed25519_keyring_tsv
      )
      mode_count = int(use_manifest_hmac_key)
      mode_count += int(use_manifest_ed25519_public_key)
      mode_count += int(use_manifest_ed25519_signer_keyring)
      if mode_count > 1:
        fail(
            "--hmac-keyring-manifest-hmac-key-file, "
            "--hmac-keyring-manifest-ed25519-public-key-file, and "
            "--hmac-keyring-manifest-ed25519-keyring-tsv are mutually exclusive"
        )
      if mode_count == 0:
        fail(
            "--hmac-keyring-manifest-json requires "
            "one of --hmac-keyring-manifest-hmac-key-file, "
            "--hmac-keyring-manifest-ed25519-public-key-file, or "
            "--hmac-keyring-manifest-ed25519-keyring-tsv"
        )
      manifest_hmac_key_bytes = None
      manifest_ed25519_public_key_path = None
      manifest_ed25519_signer_keyring = None
      if use_manifest_hmac_key:
        manifest_key_path = Path(args.hmac_keyring_manifest_hmac_key_file)
        if not manifest_key_path.is_file():
          fail(f"keyring manifest HMAC key file not found: {manifest_key_path}")
        manifest_hmac_key_bytes = manifest_key_path.read_bytes()
      elif use_manifest_ed25519_public_key:
        manifest_ed25519_public_key_path = Path(
            args.hmac_keyring_manifest_ed25519_public_key_file
        )
        if not manifest_ed25519_public_key_path.is_file():
          fail(
              "manifest Ed25519 public key file not found: "
              f"{manifest_ed25519_public_key_path}"
          )
      else:
        signer_keyring_path = Path(args.hmac_keyring_manifest_ed25519_keyring_tsv)
        if args.hmac_keyring_manifest_ed25519_keyring_sha256:
          expected_signer_keyring_hash = parse_sha256_hex(
              args.hmac_keyring_manifest_ed25519_keyring_sha256,
              "--hmac-keyring-manifest-ed25519-keyring-sha256",
          )
          if not signer_keyring_path.is_file():
            fail(f"manifest Ed25519 signer keyring file not found: {signer_keyring_path}")
          actual_signer_keyring_hash = hashlib.sha256(
              signer_keyring_path.read_bytes()
          ).hexdigest()
          if actual_signer_keyring_hash != expected_signer_keyring_hash:
            fail(
                "manifest Ed25519 signer keyring sha256 mismatch; expected "
                f"{expected_signer_keyring_hash}, got {actual_signer_keyring_hash}"
            )
        elif not signer_keyring_path.is_file():
          fail(f"manifest Ed25519 signer keyring file not found: {signer_keyring_path}")
        manifest_ed25519_signer_keyring = read_ed25519_signer_keyring(
            signer_keyring_path
        )
      expected_keyring_hash = read_keyring_manifest(
          Path(args.hmac_keyring_manifest_json),
          manifest_hmac_key_bytes=manifest_hmac_key_bytes,
          manifest_ed25519_public_key_path=manifest_ed25519_public_key_path,
          manifest_ed25519_signer_keyring=manifest_ed25519_signer_keyring,
          expected_signer_id=args.expected_keyring_signer_id,
      )
    elif args.hmac_keyring_manifest_hmac_key_file:
      fail(
          "--hmac-keyring-manifest-hmac-key-file requires "
          "--hmac-keyring-manifest-json"
      )
    elif args.hmac_keyring_manifest_ed25519_public_key_file:
      fail(
          "--hmac-keyring-manifest-ed25519-public-key-file requires "
          "--hmac-keyring-manifest-json"
      )
    elif args.hmac_keyring_manifest_ed25519_keyring_tsv:
      fail(
          "--hmac-keyring-manifest-ed25519-keyring-tsv requires "
          "--hmac-keyring-manifest-json"
      )
    elif args.hmac_keyring_manifest_ed25519_keyring_sha256:
      fail(
          "--hmac-keyring-manifest-ed25519-keyring-sha256 requires "
          "--hmac-keyring-manifest-ed25519-keyring-tsv"
      )
    elif args.expected_keyring_signer_id:
      fail("--expected-keyring-signer-id requires --hmac-keyring-manifest-json")
    if args.hmac_keyring_sha256:
      expected_hash_arg = parse_sha256_hex(
          args.hmac_keyring_sha256, "--hmac-keyring-sha256"
      )
      if expected_keyring_hash is not None and expected_hash_arg != expected_keyring_hash:
        fail(
            f"--hmac-keyring-sha256 mismatch with manifest keyring_sha256; expected "
            f"{expected_keyring_hash}, got {expected_hash_arg}"
        )
      expected_keyring_hash = expected_hash_arg
    if expected_keyring_hash is not None:
      if not keyring_path.is_file():
        fail(f"HMAC keyring file not found: {keyring_path}")
      actual_keyring_hash = hashlib.sha256(keyring_path.read_bytes()).hexdigest()
      if actual_keyring_hash != expected_keyring_hash:
        fail(
            f"HMAC keyring sha256 mismatch; expected {expected_keyring_hash}, "
            f"got {actual_keyring_hash}"
        )
    hmac_keyring = read_hmac_keyring(keyring_path)
  elif args.hmac_keyring_sha256:
    fail("--hmac-keyring-sha256 requires --hmac-keyring-tsv")
  elif args.hmac_keyring_manifest_json:
    fail("--hmac-keyring-manifest-json requires --hmac-keyring-tsv")
  elif args.hmac_keyring_manifest_hmac_key_file:
    fail("--hmac-keyring-manifest-hmac-key-file requires --hmac-keyring-manifest-json")
  elif args.hmac_keyring_manifest_ed25519_public_key_file:
    fail(
        "--hmac-keyring-manifest-ed25519-public-key-file requires "
        "--hmac-keyring-manifest-json"
    )
  elif args.hmac_keyring_manifest_ed25519_keyring_tsv:
    fail(
        "--hmac-keyring-manifest-ed25519-keyring-tsv requires "
        "--hmac-keyring-manifest-json"
    )
  elif args.hmac_keyring_manifest_ed25519_keyring_sha256:
    fail(
        "--hmac-keyring-manifest-ed25519-keyring-sha256 requires "
        "--hmac-keyring-manifest-ed25519-keyring-tsv"
    )
  elif args.expected_keyring_signer_id:
    fail("--expected-keyring-signer-id requires --hmac-keyring-manifest-json")
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

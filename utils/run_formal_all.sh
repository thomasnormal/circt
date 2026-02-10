#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
run_formal_all.sh [options]

Runs formal BMC/LEC suites and summarizes results.

Options:
  --out-dir DIR          Output directory for logs/results (default: formal-results-YYYYMMDD)
  --sv-tests DIR         sv-tests root (default: ~/sv-tests)
  --verilator DIR        verilator-verification root (default: ~/verilator-verification)
  --yosys DIR            yosys/tests/sva root (default: ~/yosys/tests/sva)
  --z3-bin PATH          Path to z3 binary (optional)
  --baseline-file FILE   Baseline TSV file (default: utils/formal-baselines.tsv)
  --plan-file FILE       Project plan file to update (default: PROJECT_PLAN.md)
  --update-baselines     Update baseline file and PROJECT_PLAN.md table
  --fail-on-diff         Fail if results differ from baseline file
  --strict-gate          Fail on new fail/error/xpass and pass-rate regression vs baseline
  --baseline-window N    Baseline rows per suite/mode used for gate comparison
                         (default: 1, latest baseline only)
  --baseline-window-days N
                         Limit baseline comparison to rows within N days of the
                         suite/mode latest baseline date (default: 0, disabled)
  --fail-on-new-xpass    Fail when xpass count increases vs baseline
  --fail-on-passrate-regression
                         Fail when pass-rate decreases vs baseline
  --fail-on-new-failure-cases
                         Fail when fail-like case IDs increase vs baseline
  --fail-on-new-bmc-timeout-cases
                         Fail when BMC timeout-case count increases vs baseline
  --fail-on-new-bmc-unknown-cases
                         Fail when BMC unknown-case count increases vs baseline
  --fail-on-new-e2e-mode-diff-strict-only-fail
                         Fail when OpenTitan E2E mode-diff strict_only_fail
                         count increases vs baseline
  --fail-on-new-e2e-mode-diff-status-diff
                         Fail when OpenTitan E2E mode-diff status_diff
                         count increases vs baseline
  --fail-on-new-e2e-mode-diff-strict-only-pass
                         Fail when OpenTitan E2E mode-diff strict_only_pass
                         count increases vs baseline
  --fail-on-new-e2e-mode-diff-missing-in-e2e
                         Fail when OpenTitan E2E mode-diff missing_in_e2e
                         count increases vs baseline
  --fail-on-new-e2e-mode-diff-missing-in-e2e-strict
                         Fail when OpenTitan E2E mode-diff missing_in_e2e_strict
                         count increases vs baseline
  --fail-on-new-opentitan-lec-strict-xprop-counter KEY
                         Fail when OpenTitan strict LEC summary counter KEY
                         increases vs baseline (repeatable)
  --expected-failures-file FILE
                         TSV with suite/mode expected fail+error budgets
  --expectations-dry-run
                         Preview expectation refresh/prune without rewriting files
  --expectations-dry-run-report-jsonl FILE
                         Append dry-run operation summaries as JSON Lines
  --expectations-dry-run-report-max-sample-rows N
                         Max sampled rows embedded per dry-run JSONL operation
                         (default: 5; 0 disables row samples)
  --expectations-dry-run-report-hmac-key-file FILE
                         Optional key file for HMAC-SHA256 signing of run_end
                         payload digest
  --expectations-dry-run-report-hmac-key-id ID
                         Optional key identifier emitted in run_meta/run_end
                         when HMAC signing is enabled
  --fail-on-unexpected-failures
                         Fail when fail/error exceed expected-failure budgets
  --fail-on-unused-expected-failures
                         Fail when expected-failures rows are unused
  --prune-expected-failures-file FILE
                         Rewrite expected-failures file by pruning stale rows
  --prune-expected-failures-drop-unused
                         Drop expected-failures rows unused in current run
  --refresh-expected-failures-file FILE
                         Rewrite expected-failures TSV from current run
  --refresh-expected-failures-include-suite-regex REGEX
                         Refresh only suite rows matching REGEX
  --refresh-expected-failures-include-mode-regex REGEX
                         Refresh only mode rows matching REGEX
  --expected-failure-cases-file FILE
                         TSV with expected failing test cases (suite/mode/id)
                         id_kind supports:
                         base|base_diag|path|aggregate|base_regex|base_diag_regex|path_regex
  --fail-on-unexpected-failure-cases
                         Fail when observed failing cases are not expected
  --fail-on-expired-expected-failure-cases
                         Fail when any expected case is expired by expires_on
  --fail-on-unmatched-expected-failure-cases
                         Fail when expected cases have no observed match
  --prune-expected-failure-cases-file FILE
                         Rewrite expected-cases file by pruning stale rows
  --prune-expected-failure-cases-drop-unmatched
                         Drop expected-case rows with matched_count=0
  --prune-expected-failure-cases-drop-expired
                         Drop expected-case rows with expired=yes
  --refresh-expected-failure-cases-file FILE
                         Rewrite expected-failure-cases TSV from current run
  --refresh-expected-failure-cases-default-expires-on YYYY-MM-DD
                         Default expires_on for newly added refreshed case rows
  --refresh-expected-failure-cases-collapse-status-any
                         Collapse refreshed case statuses to ANY per case key
  --refresh-expected-failure-cases-include-suite-regex REGEX
                         Refresh only case rows with suite matching REGEX
  --refresh-expected-failure-cases-include-mode-regex REGEX
                         Refresh only case rows with mode matching REGEX
  --refresh-expected-failure-cases-include-status-regex REGEX
                         Refresh only case rows with status matching REGEX
  --refresh-expected-failure-cases-include-id-regex REGEX
                         Refresh only case rows with id matching REGEX
  --json-summary FILE    Write machine-readable JSON summary (default: <out-dir>/summary.json)
  --lane-state-tsv FILE  Persistent lane state TSV (for resumable matrix runs)
  --resume-from-lane-state
                         Reuse completed lane rows from --lane-state-tsv
  --reset-lane-state     Reset/truncate --lane-state-tsv before running
  --merge-lane-state-tsv FILE
                         Merge lane-state rows from FILE (repeatable)
  --lane-state-hmac-key-file FILE
                         HMAC key file used to sign and verify lane-state
                         manifests (<lane-state>.manifest.json)
  --lane-state-hmac-keyring-tsv FILE
                         HMAC keyring TSV used to resolve lane-state signing
                         keys by key-id (<key_id>\t<key_file>\t...)
  --lane-state-hmac-keyring-sha256 HEX
                         Optional SHA256 pin for --lane-state-hmac-keyring-tsv
  --lane-state-hmac-key-id ID
                         Optional key identifier embedded in lane-state
                         manifests; verified on resume/merge when set
  --lane-state-manifest-ed25519-private-key-file FILE
                         Ed25519 private key for lane-state manifest signing
  --lane-state-manifest-ed25519-public-key-file FILE
                         Ed25519 public key for lane-state manifest verification
  --lane-state-manifest-ed25519-keyring-tsv FILE
                         Ed25519 public-key keyring TSV for lane-state manifest
                         verification (columns: key_id, public_key_file_path,
                         not_before, not_after, status, key_sha256)
  --lane-state-manifest-ed25519-keyring-sha256 HEX
                         Optional SHA256 pin for
                         --lane-state-manifest-ed25519-keyring-tsv
  --lane-state-manifest-ed25519-ca-file FILE
                         Optional CA/trust-anchor PEM for Ed25519 keyring cert
                         verification
  --lane-state-manifest-ed25519-crl-file FILE
                         Optional CRL PEM checked with --lane-state-manifest-ed25519-ca-file
                         during Ed25519 keyring cert verification
  --lane-state-manifest-ed25519-crl-refresh-cmd CMD
                         Optional command run before keyring verification to
                         refresh --lane-state-manifest-ed25519-crl-file
  --lane-state-manifest-ed25519-crl-refresh-uri URI
                         Optional built-in fetch URI (file/http/https) used to
                         refresh --lane-state-manifest-ed25519-crl-file
  --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp
                         Resolve CRL refresh URI from selected key certificate
                         CRL Distribution Points (keyring mode)
  --lane-state-manifest-ed25519-crl-refresh-auto-uri-policy MODE
                         URI selection policy for cert CDP auto mode:
                         first | last | require_single
  --lane-state-manifest-ed25519-crl-refresh-auto-uri-allowed-schemes LIST
                         Allowed URI schemes for cert CDP auto mode
                         (comma-separated subset of file,http,https)
  --lane-state-manifest-ed25519-refresh-auto-uri-policy MODE
                         Shared default URI selection policy for cert-driven
                         CRL/OCSP auto modes (specific per-artifact flags
                         override this): first | last | require_single
  --lane-state-manifest-ed25519-refresh-auto-uri-allowed-schemes LIST
                         Shared default allowed URI schemes for cert-driven
                         CRL/OCSP auto modes (specific per-artifact flags
                         override this): comma-separated subset of
                         file,http,https
  --lane-state-manifest-ed25519-refresh-policy-profiles-json FILE
                         JSON policy profile registry for cert-driven refresh
                         auto-URI defaults (schema_version=1)
  --lane-state-manifest-ed25519-refresh-policy-profile NAME
                         Profile name selected from
                         --lane-state-manifest-ed25519-refresh-policy-profiles-json
  --lane-state-manifest-ed25519-refresh-policy-profiles-sha256 HEX
                         Optional SHA256 pin for
                         --lane-state-manifest-ed25519-refresh-policy-profiles-json
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-json FILE
                         Optional signed manifest for refresh policy profiles
                         integrity metadata
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-public-key-file FILE
                         Ed25519 public key used to verify
                         --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-json
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv FILE
                         Ed25519 keyring TSV used to resolve signer key_id for
                         --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-json
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-sha256 HEX
                         Optional SHA256 pin for
                         --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file FILE
                         Optional CA/trust-anchor PEM used to verify signer
                         certificates referenced by manifest signer keyring rows
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-crl-file FILE
                         Optional CRL PEM for signer-certificate revocation and
                         freshness checks in manifest signer keyring mode
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file FILE
                         Optional DER OCSP response for signer-certificate
                         status checks in manifest signer keyring mode
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-sha256 HEX
                         Optional SHA256 pin for
                         --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-max-age-secs N
                         Maximum OCSP response age from thisUpdate in seconds
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-next-update
                         Require Next Update field in signer-certificate OCSP
                         response for manifest signer keyring mode
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-file FILE
                         Optional responder certificate PEM used to pin signer
                         OCSP responder identity in manifest signer keyring mode
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-issuer-cert-file FILE
                         Optional issuer certificate PEM used for signer OCSP
                         --issuer selection (defaults to manifest keyring CA)
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-sha256 HEX
                         Optional SHA256 pin for signer OCSP responder cert
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-responder-ocsp-signing
                         Require signer OCSP responder cert EKU to include OCSP
                         Signing
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-responder-aki-match-ca-ski
                         Require signer OCSP responder cert AKI keyid to match
                         manifest keyring CA cert SKI
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-id-regex REGEX
                         Optional regex constraint applied to signer OCSP
                         Responder Id in manifest signer keyring mode
  --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-key-id ID
                         Optional expected key_id for profile manifest signer
  --lane-state-manifest-ed25519-crl-refresh-retries N
                         Retry count for
                         --lane-state-manifest-ed25519-crl-refresh-cmd or
                         --lane-state-manifest-ed25519-crl-refresh-uri or
                         --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp
  --lane-state-manifest-ed25519-crl-refresh-delay-secs N
                         Delay between CRL refresh retries in seconds
  --lane-state-manifest-ed25519-crl-refresh-timeout-secs N
                         Timeout per CRL refresh attempt in seconds
  --lane-state-manifest-ed25519-crl-refresh-jitter-secs N
                         Additional random delay per retry attempt
                         (0..N seconds) for CRL refresh
  --lane-state-manifest-ed25519-crl-refresh-metadata-file FILE
                         Optional JSON object file emitted/updated by CRL
                         refresh command and embedded in signed refresh
                         provenance
  --lane-state-manifest-ed25519-crl-refresh-metadata-require-transport MODE
                         Require CRL refresh metadata `transport` value
  --lane-state-manifest-ed25519-crl-refresh-metadata-require-status STATUS
                         Require CRL refresh metadata `status` value
  --lane-state-manifest-ed25519-crl-refresh-metadata-require-uri-regex REGEX
                         Require CRL refresh metadata `uri` to match REGEX
  --lane-state-manifest-ed25519-crl-refresh-metadata-require-tls-peer-sha256 HEX
                         Require CRL refresh metadata `tls_peer_sha256` value
  --lane-state-manifest-ed25519-crl-refresh-metadata-require-cert-chain-sha256 HEX
                         Require CRL refresh metadata `cert_chain_sha256` to
                         contain HEX digest
  --lane-state-manifest-ed25519-crl-refresh-metadata-require-artifact-sha256
                         Require CRL refresh metadata `artifact_sha256` to
                         match the refreshed CRL artifact digest
  --lane-state-manifest-ed25519-crl-refresh-metadata-require-ca-cert-in-cert-chain
                         Require CRL refresh metadata `cert_chain_sha256` to
                         contain configured CA cert digest
  --lane-state-manifest-ed25519-crl-refresh-metadata-require-tls-peer-in-cert-chain
                         Require CRL refresh metadata `tls_peer_sha256` to be
                         present in `cert_chain_sha256` (https transport)
  --lane-state-manifest-ed25519-crl-refresh-metadata-require-cert-chain-length-min N
                         Require CRL refresh metadata `cert_chain_sha256` to
                         contain at least N certificates
  --lane-state-manifest-ed25519-crl-refresh-metadata-max-age-secs N
                         Require CRL refresh metadata `fetched_at_utc` age
                         not older than N seconds
  --lane-state-manifest-ed25519-crl-refresh-metadata-max-future-skew-secs N
                         Require CRL refresh metadata `fetched_at_utc` not
                         more than N seconds in the future
  --lane-state-manifest-ed25519-ocsp-response-file FILE
                         Optional DER OCSP response checked with
                         --lane-state-manifest-ed25519-ca-file during Ed25519
                         keyring cert verification
  --lane-state-manifest-ed25519-ocsp-refresh-cmd CMD
                         Optional command run before keyring verification to
                         refresh --lane-state-manifest-ed25519-ocsp-response-file
  --lane-state-manifest-ed25519-ocsp-refresh-uri URI
                         Optional built-in fetch URI (file/http/https) used to
                         refresh --lane-state-manifest-ed25519-ocsp-response-file
  --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia
                         Resolve OCSP refresh URI from selected key certificate
                         Authority Information Access (keyring mode)
  --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-policy MODE
                         URI selection policy for cert AIA auto mode:
                         first | last | require_single
  --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-allowed-schemes LIST
                         Allowed URI schemes for cert AIA auto mode
                         (comma-separated subset of file,http,https)
  --lane-state-manifest-ed25519-ocsp-refresh-retries N
                         Retry count for
                         --lane-state-manifest-ed25519-ocsp-refresh-cmd or
                         --lane-state-manifest-ed25519-ocsp-refresh-uri or
                         --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia
  --lane-state-manifest-ed25519-ocsp-refresh-delay-secs N
                         Delay between OCSP refresh retries in seconds
  --lane-state-manifest-ed25519-ocsp-refresh-timeout-secs N
                         Timeout per OCSP refresh attempt in seconds
  --lane-state-manifest-ed25519-ocsp-refresh-jitter-secs N
                         Additional random delay per retry attempt
                         (0..N seconds) for OCSP refresh
  --lane-state-manifest-ed25519-ocsp-refresh-metadata-file FILE
                         Optional JSON object file emitted/updated by OCSP
                         refresh command and embedded in signed refresh
                         provenance
  --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-transport MODE
                         Require OCSP refresh metadata `transport` value
  --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-status STATUS
                         Require OCSP refresh metadata `status` value
  --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-uri-regex REGEX
                         Require OCSP refresh metadata `uri` to match REGEX
  --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-tls-peer-sha256 HEX
                         Require OCSP refresh metadata `tls_peer_sha256` value
  --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-cert-chain-sha256 HEX
                         Require OCSP refresh metadata `cert_chain_sha256` to
                         contain HEX digest
  --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-artifact-sha256
                         Require OCSP refresh metadata `artifact_sha256` to
                         match the refreshed OCSP artifact digest
  --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-ca-cert-in-cert-chain
                         Require OCSP refresh metadata `cert_chain_sha256` to
                         contain configured CA cert digest
  --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-tls-peer-in-cert-chain
                         Require OCSP refresh metadata `tls_peer_sha256` to be
                         present in `cert_chain_sha256` (https transport)
  --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-cert-chain-length-min N
                         Require OCSP refresh metadata `cert_chain_sha256` to
                         contain at least N certificates
  --lane-state-manifest-ed25519-ocsp-refresh-metadata-max-age-secs N
                         Require OCSP refresh metadata `fetched_at_utc` age
                         not older than N seconds
  --lane-state-manifest-ed25519-ocsp-refresh-metadata-max-future-skew-secs N
                         Require OCSP refresh metadata `fetched_at_utc` not
                         more than N seconds in the future
  --lane-state-manifest-ed25519-ocsp-response-sha256 HEX
                         Optional SHA256 pin for
                         --lane-state-manifest-ed25519-ocsp-response-file
  --lane-state-manifest-ed25519-ocsp-responder-cert-file FILE
                         Optional responder certificate PEM used to pin OCSP
                         signer identity
  --lane-state-manifest-ed25519-ocsp-issuer-cert-file FILE
                         Optional issuer certificate PEM used for OCSP
                         --issuer selection (defaults to
                         --lane-state-manifest-ed25519-ca-file)
  --lane-state-manifest-ed25519-ocsp-responder-cert-sha256 HEX
                         Optional SHA256 pin for
                         --lane-state-manifest-ed25519-ocsp-responder-cert-file
  --lane-state-manifest-ed25519-ocsp-require-responder-ocsp-signing
                         Require responder cert EKU to include OCSP Signing
  --lane-state-manifest-ed25519-ocsp-require-responder-aki-match-ca-ski
                         Require responder cert AKI keyid to match CA cert SKI
  --lane-state-manifest-ed25519-ocsp-max-age-secs N
                         Max allowed OCSP response age in seconds from
                         thisUpdate (default when OCSP is enabled: 604800)
  --lane-state-manifest-ed25519-ocsp-require-next-update
                         Require OCSP responses to include Next Update
  --lane-state-manifest-ed25519-ocsp-responder-id-regex REGEX
                         Optional regex constraint applied to OCSP Responder Id
  --lane-state-manifest-ed25519-cert-subject-regex REGEX
                         Optional regex constraint applied to selected Ed25519
                         certificate subject (keyring mode)
  --lane-state-manifest-ed25519-key-id ID
                         Optional key identifier embedded in Ed25519 manifests
  --include-lane-regex REGEX
                         Run only lanes whose lane-id matches REGEX
  --exclude-lane-regex REGEX
                         Skip lanes whose lane-id matches REGEX
  --bmc-run-smtlib        Use circt-bmc --run-smtlib (external z3) in suite runs
  --bmc-allow-multi-clock Add --allow-multi-clock to BMC runs
  --bmc-assume-known-inputs  Add --assume-known-inputs to BMC runs
  --lec-assume-known-inputs  Add --assume-known-inputs to LEC runs
  --lec-accept-xprop-only    Treat XPROP_ONLY mismatches as equivalent in LEC runs
  --with-opentitan       Run OpenTitan LEC script
  --with-opentitan-lec-strict
                         Run strict OpenTitan LEC lane (LEC_X_OPTIMISTIC=0)
  --with-opentitan-e2e   Run OpenTitan non-smoke E2E parity lane
  --with-opentitan-e2e-strict
                         Run strict OpenTitan non-smoke E2E audit lane
                         (LEC strict-x)
  --opentitan DIR        OpenTitan root (default: ~/opentitan)
  --opentitan-lec-impl-filter REGEX
                         Regex filter for OpenTitan LEC implementations
  --opentitan-lec-include-masked
                         Include masked OpenTitan LEC implementations
  --opentitan-lec-strict-dump-unknown-sources
                         Enable strict OpenTitan LEC unknown-source dumping
                         (`--dump-unknown-sources`) for X-prop triage
  --opentitan-e2e-sim-targets LIST
                         Comma-separated OpenTitan E2E sim targets
  --opentitan-e2e-verilog-targets LIST
                         Comma-separated OpenTitan E2E parse targets
  --opentitan-e2e-sim-timeout SECS
                         OpenTitan E2E sim timeout per target
  --opentitan-e2e-impl-filter REGEX
                         OpenTitan E2E LEC implementation regex filter
  --opentitan-e2e-include-masked
                         Include masked OpenTitan E2E LEC implementations
  --opentitan-e2e-lec-x-optimistic
                         Force OpenTitan E2E LEC to x-optimistic mode
  --opentitan-e2e-lec-strict-x
                         Force OpenTitan E2E LEC to strict non-optimistic mode
  --circt-verilog PATH   Path to circt-verilog (default: <repo>/build/bin/circt-verilog)
  --circt-verilog-avip PATH
                         Path override for AVIP runs (default: --circt-verilog value)
  --circt-verilog-opentitan PATH
                         Path override for OpenTitan runs (default: --circt-verilog value)
  --with-avip            Run AVIP compile smoke using run_avip_circt_verilog.sh
  --avip-glob GLOB       Glob for AVIP dirs (default: ~/mbit/*avip*)
  -h, --help             Show this help
USAGE
}

normalize_auto_uri_allowed_schemes() {
  local raw_value="$1"
  local option_name="$2"
  local token=""
  local normalized=""
  local seen=","
  local IFS=','

  read -r -a _scheme_tokens <<< "$raw_value"
  if [[ "${#_scheme_tokens[@]}" -eq 0 ]]; then
    echo "invalid ${option_name}: expected comma-separated subset of file,http,https" >&2
    exit 1
  fi
  for token in "${_scheme_tokens[@]}"; do
    token="${token#"${token%%[![:space:]]*}"}"
    token="${token%"${token##*[![:space:]]}"}"
    token="${token,,}"
    if [[ -z "$token" || ! "$token" =~ ^(file|http|https)$ ]]; then
      echo "invalid ${option_name}: expected comma-separated subset of file,http,https" >&2
      exit 1
    fi
    if [[ "$seen" == *",$token,"* ]]; then
      continue
    fi
    if [[ -n "$normalized" ]]; then
      normalized+=","
    fi
    normalized+="$token"
    seen+="${token},"
  done
  if [[ -z "$normalized" ]]; then
    echo "invalid ${option_name}: expected comma-separated subset of file,http,https" >&2
    exit 1
  fi
  printf '%s\n' "$normalized"
}

parse_refresh_auto_uri_policy_profile() {
  local profiles_json="$1"
  local profile_name="$2"
  python3 - "$profiles_json" "$profile_name" <<'PY'
import json
import sys
from pathlib import Path

profiles_path = Path(sys.argv[1])
profile_name = sys.argv[2].strip()

try:
  payload = json.loads(profiles_path.read_text(encoding="utf-8"))
except Exception:
  print(
      f"invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json: file '{profiles_path}' must contain JSON object",
      file=sys.stderr,
  )
  raise SystemExit(1)

if not isinstance(payload, dict) or not payload:
  print(
      f"invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json: file '{profiles_path}' must contain non-empty JSON object",
      file=sys.stderr,
  )
  raise SystemExit(1)

unknown_top_keys = sorted(set(payload.keys()) - {"schema_version", "profiles"})
if unknown_top_keys:
  print(
      f"invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json: unknown key '{unknown_top_keys[0]}'",
      file=sys.stderr,
  )
  raise SystemExit(1)

schema_version = payload.get("schema_version")
if schema_version != 1:
  print(
      "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json: schema_version must be 1",
      file=sys.stderr,
  )
  raise SystemExit(1)

profiles = payload.get("profiles")
if not isinstance(profiles, dict) or not profiles:
  print(
      "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json: profiles must be non-empty object",
      file=sys.stderr,
  )
  raise SystemExit(1)

profile = profiles.get(profile_name)
if profile is None:
  print(
      f"invalid --lane-state-manifest-ed25519-refresh-policy-profile: unknown profile '{profile_name}'",
      file=sys.stderr,
  )
  raise SystemExit(1)
if not isinstance(profile, dict):
  print(
      f"invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json.profiles.{profile_name}: expected object",
      file=sys.stderr,
  )
  raise SystemExit(1)

def emit_if_string(scope: dict, scope_name: str, key: str, out_key: str):
  if key not in scope:
    return
  value = scope[key]
  if not isinstance(value, str) or not value.strip():
    print(
      f"invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json.{scope_name}.{key}: expected non-empty string",
      file=sys.stderr,
    )
    raise SystemExit(1)
  print(f"{out_key}\t{value.strip()}")

def emit_if_bool(scope: dict, scope_name: str, key: str, out_key: str):
  if key not in scope:
    return
  value = scope[key]
  if not isinstance(value, bool):
    print(
      f"invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json.{scope_name}.{key}: expected boolean",
      file=sys.stderr,
    )
    raise SystemExit(1)
  print(f"{out_key}\t{1 if value else 0}")

def emit_if_nonneg_int(scope: dict, scope_name: str, key: str, out_key: str):
  if key not in scope:
    return
  value = scope[key]
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    print(
      f"invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json.{scope_name}.{key}: expected non-negative integer",
      file=sys.stderr,
    )
    raise SystemExit(1)
  print(f"{out_key}\t{value}")

unknown_profile_keys = sorted(
  set(profile.keys()) - {
      "auto_uri_policy",
      "auto_uri_allowed_schemes",
      "refresh_retries",
      "refresh_delay_secs",
      "refresh_timeout_secs",
      "refresh_jitter_secs",
      "refresh_metadata_require_transport",
      "refresh_metadata_require_status",
      "refresh_metadata_require_uri_regex",
      "refresh_metadata_require_tls_peer_sha256",
      "refresh_metadata_require_cert_chain_sha256",
      "refresh_metadata_require_artifact_sha256",
      "refresh_metadata_require_cert_chain_length_min",
      "refresh_metadata_require_ca_cert_in_cert_chain",
      "refresh_metadata_require_tls_peer_in_cert_chain",
      "refresh_metadata_max_age_secs",
      "refresh_metadata_max_future_skew_secs",
      "crl",
      "ocsp",
  }
)
if unknown_profile_keys:
  print(
      f"invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json.profiles.{profile_name}: unknown key '{unknown_profile_keys[0]}'",
      file=sys.stderr,
  )
  raise SystemExit(1)

emit_if_string(profile, f"profiles.{profile_name}", "auto_uri_policy", "shared_auto_uri_policy")
emit_if_string(profile, f"profiles.{profile_name}", "auto_uri_allowed_schemes", "shared_auto_uri_allowed_schemes")
emit_if_nonneg_int(
    profile,
    f"profiles.{profile_name}",
    "refresh_retries",
    "shared_refresh_retries",
)
emit_if_nonneg_int(
    profile,
    f"profiles.{profile_name}",
    "refresh_delay_secs",
    "shared_refresh_delay_secs",
)
emit_if_nonneg_int(
    profile,
    f"profiles.{profile_name}",
    "refresh_timeout_secs",
    "shared_refresh_timeout_secs",
)
emit_if_nonneg_int(
    profile,
    f"profiles.{profile_name}",
    "refresh_jitter_secs",
    "shared_refresh_jitter_secs",
)
emit_if_string(
    profile,
    f"profiles.{profile_name}",
    "refresh_metadata_require_transport",
    "shared_refresh_metadata_require_transport",
)
emit_if_string(
    profile,
    f"profiles.{profile_name}",
    "refresh_metadata_require_status",
    "shared_refresh_metadata_require_status",
)
emit_if_string(
    profile,
    f"profiles.{profile_name}",
    "refresh_metadata_require_uri_regex",
    "shared_refresh_metadata_require_uri_regex",
)
emit_if_string(
    profile,
    f"profiles.{profile_name}",
    "refresh_metadata_require_tls_peer_sha256",
    "shared_refresh_metadata_require_tls_peer_sha256",
)
emit_if_string(
    profile,
    f"profiles.{profile_name}",
    "refresh_metadata_require_cert_chain_sha256",
    "shared_refresh_metadata_require_cert_chain_sha256",
)
emit_if_bool(
    profile,
    f"profiles.{profile_name}",
    "refresh_metadata_require_artifact_sha256",
    "shared_refresh_metadata_require_artifact_sha256",
)
emit_if_nonneg_int(
    profile,
    f"profiles.{profile_name}",
    "refresh_metadata_require_cert_chain_length_min",
    "shared_refresh_metadata_require_cert_chain_length_min",
)
emit_if_bool(
    profile,
    f"profiles.{profile_name}",
    "refresh_metadata_require_ca_cert_in_cert_chain",
    "shared_refresh_metadata_require_ca_cert_in_cert_chain",
)
emit_if_bool(
    profile,
    f"profiles.{profile_name}",
    "refresh_metadata_require_tls_peer_in_cert_chain",
    "shared_refresh_metadata_require_tls_peer_in_cert_chain",
)
emit_if_nonneg_int(
    profile,
    f"profiles.{profile_name}",
    "refresh_metadata_max_age_secs",
    "shared_refresh_metadata_max_age_secs",
)
emit_if_nonneg_int(
    profile,
    f"profiles.{profile_name}",
    "refresh_metadata_max_future_skew_secs",
    "shared_refresh_metadata_max_future_skew_secs",
)

for artifact in ("crl", "ocsp"):
  section = profile.get(artifact)
  if section is None:
    continue
  if not isinstance(section, dict):
    print(
        f"invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json.profiles.{profile_name}.{artifact}: expected object",
        file=sys.stderr,
    )
    raise SystemExit(1)
  unknown_section_keys = sorted(
      set(section.keys())
      - {
          "auto_uri_policy",
          "auto_uri_allowed_schemes",
          "refresh_retries",
          "refresh_delay_secs",
          "refresh_timeout_secs",
          "refresh_jitter_secs",
          "refresh_metadata_require_transport",
          "refresh_metadata_require_status",
          "refresh_metadata_require_uri_regex",
          "refresh_metadata_require_tls_peer_sha256",
          "refresh_metadata_require_cert_chain_sha256",
          "refresh_metadata_require_artifact_sha256",
          "refresh_metadata_require_cert_chain_length_min",
          "refresh_metadata_require_ca_cert_in_cert_chain",
          "refresh_metadata_require_tls_peer_in_cert_chain",
          "refresh_metadata_max_age_secs",
          "refresh_metadata_max_future_skew_secs",
      }
  )
  if unknown_section_keys:
    print(
        f"invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json.profiles.{profile_name}.{artifact}: unknown key '{unknown_section_keys[0]}'",
        file=sys.stderr,
    )
    raise SystemExit(1)
  emit_if_string(section, f"profiles.{profile_name}.{artifact}", "auto_uri_policy", f"{artifact}_auto_uri_policy")
  emit_if_string(section, f"profiles.{profile_name}.{artifact}", "auto_uri_allowed_schemes", f"{artifact}_auto_uri_allowed_schemes")
  emit_if_nonneg_int(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_retries",
      f"{artifact}_refresh_retries",
  )
  emit_if_nonneg_int(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_delay_secs",
      f"{artifact}_refresh_delay_secs",
  )
  emit_if_nonneg_int(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_timeout_secs",
      f"{artifact}_refresh_timeout_secs",
  )
  emit_if_nonneg_int(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_jitter_secs",
      f"{artifact}_refresh_jitter_secs",
  )
  emit_if_string(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_metadata_require_transport",
      f"{artifact}_refresh_metadata_require_transport",
  )
  emit_if_string(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_metadata_require_status",
      f"{artifact}_refresh_metadata_require_status",
  )
  emit_if_string(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_metadata_require_uri_regex",
      f"{artifact}_refresh_metadata_require_uri_regex",
  )
  emit_if_string(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_metadata_require_tls_peer_sha256",
      f"{artifact}_refresh_metadata_require_tls_peer_sha256",
  )
  emit_if_string(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_metadata_require_cert_chain_sha256",
      f"{artifact}_refresh_metadata_require_cert_chain_sha256",
  )
  emit_if_bool(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_metadata_require_artifact_sha256",
      f"{artifact}_refresh_metadata_require_artifact_sha256",
  )
  emit_if_nonneg_int(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_metadata_require_cert_chain_length_min",
      f"{artifact}_refresh_metadata_require_cert_chain_length_min",
  )
  emit_if_bool(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_metadata_require_ca_cert_in_cert_chain",
      f"{artifact}_refresh_metadata_require_ca_cert_in_cert_chain",
  )
  emit_if_bool(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_metadata_require_tls_peer_in_cert_chain",
      f"{artifact}_refresh_metadata_require_tls_peer_in_cert_chain",
  )
  emit_if_nonneg_int(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_metadata_max_age_secs",
      f"{artifact}_refresh_metadata_max_age_secs",
  )
  emit_if_nonneg_int(
      section,
      f"profiles.{profile_name}.{artifact}",
      "refresh_metadata_max_future_skew_secs",
      f"{artifact}_refresh_metadata_max_future_skew_secs",
  )
PY
}

verify_refresh_policy_profiles_manifest() {
  local manifest_json="$1"
  local expected_profiles_sha="$2"
  local public_key_file="$3"
  local expected_key_id="$4"
  python3 - "$manifest_json" "$expected_profiles_sha" "$public_key_file" "$expected_key_id" <<'PY'
import base64
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected_profiles_sha = sys.argv[2].strip()
public_key_file = Path(sys.argv[3])
expected_key_id = sys.argv[4].strip()

try:
  payload = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception as ex:
  print(
      f"invalid refresh policy profiles manifest: unable to parse '{manifest_path}' ({ex})",
      file=sys.stderr,
  )
  raise SystemExit(1)

if not isinstance(payload, dict):
  print("invalid refresh policy profiles manifest: expected JSON object", file=sys.stderr)
  raise SystemExit(1)

unknown_keys = sorted(
    set(payload.keys())
    - {
        "schema_version",
        "generated_at_utc",
        "profiles_sha256",
        "signature_mode",
        "key_id",
        "signature_ed25519_base64",
    }
)
if unknown_keys:
  print(
      f"invalid refresh policy profiles manifest: unknown key '{unknown_keys[0]}'",
      file=sys.stderr,
  )
  raise SystemExit(1)

if payload.get("schema_version") != 1:
  print("invalid refresh policy profiles manifest: schema_version must be 1", file=sys.stderr)
  raise SystemExit(1)

generated_at = payload.get("generated_at_utc")
if generated_at is not None:
  if not isinstance(generated_at, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", generated_at):
    print(
        "invalid refresh policy profiles manifest: generated_at_utc must be UTC RFC3339 timestamp",
        file=sys.stderr,
    )
    raise SystemExit(1)

profiles_sha256 = payload.get("profiles_sha256")
if not isinstance(profiles_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", profiles_sha256):
  print("invalid refresh policy profiles manifest: profiles_sha256 must be 64 hex chars", file=sys.stderr)
  raise SystemExit(1)
if profiles_sha256 != expected_profiles_sha:
  print(
      f"invalid refresh policy profiles manifest: profiles_sha256 mismatch (expected {expected_profiles_sha}, found {profiles_sha256})",
      file=sys.stderr,
  )
  raise SystemExit(1)

signature_mode = payload.get("signature_mode")
if signature_mode != "ed25519":
  print(
      f"invalid refresh policy profiles manifest: signature_mode mismatch (expected ed25519, found {signature_mode})",
      file=sys.stderr,
  )
  raise SystemExit(1)

manifest_key_id = payload.get("key_id", "")
if manifest_key_id and (not isinstance(manifest_key_id, str) or not manifest_key_id.strip()):
  print("invalid refresh policy profiles manifest: key_id must be non-empty string", file=sys.stderr)
  raise SystemExit(1)
manifest_key_id = manifest_key_id.strip() if isinstance(manifest_key_id, str) else ""
if expected_key_id and manifest_key_id != expected_key_id:
  print(
      f"invalid refresh policy profiles manifest: key_id mismatch (expected '{expected_key_id}', found '{manifest_key_id}')",
      file=sys.stderr,
  )
  raise SystemExit(1)

signature_b64 = payload.get("signature_ed25519_base64")
if not isinstance(signature_b64, str) or not signature_b64.strip():
  print(
      "invalid refresh policy profiles manifest: signature_ed25519_base64 must be non-empty string",
      file=sys.stderr,
  )
  raise SystemExit(1)
try:
  signature_bytes = base64.b64decode(signature_b64, validate=True)
except Exception:
  print(
      "invalid refresh policy profiles manifest: signature_ed25519_base64 must be valid base64",
      file=sys.stderr,
  )
  raise SystemExit(1)

sig_payload = dict(payload)
del sig_payload["signature_ed25519_base64"]
canonical_payload = json.dumps(sig_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

with tempfile.NamedTemporaryFile(delete=False) as payload_file:
  payload_file.write(canonical_payload)
  payload_path = payload_file.name
with tempfile.NamedTemporaryFile(delete=False) as signature_file:
  signature_file.write(signature_bytes)
  signature_path = signature_file.name
try:
  verify_result = subprocess.run(
      [
          "openssl",
          "pkeyutl",
          "-verify",
          "-pubin",
          "-inkey",
          str(public_key_file),
          "-sigfile",
          signature_path,
          "-rawin",
          "-in",
          payload_path,
      ],
      check=False,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  )
finally:
  Path(payload_path).unlink(missing_ok=True)
  Path(signature_path).unlink(missing_ok=True)

if verify_result.returncode != 0:
  stderr = verify_result.stderr.decode("utf-8", errors="replace").strip()
  stdout = verify_result.stdout.decode("utf-8", errors="replace").strip()
  detail = stderr or stdout or "openssl verify failed"
  print(f"invalid refresh policy profiles manifest: signature verification failed ({detail})", file=sys.stderr)
  raise SystemExit(1)

print(f"manifest_key_id\t{manifest_key_id}")
PY
}

resolve_refresh_policy_profiles_manifest_public_key_from_keyring() {
  local manifest_json="$1"
  local keyring_tsv="$2"
  local expected_keyring_sha="$3"
  local expected_key_id="$4"
  local ca_file="$5"
  local crl_file="$6"
  local ocsp_response_file="$7"
  local expected_ocsp_response_sha="$8"
  local ocsp_max_age_secs="$9"
  local ocsp_require_next_update="${10}"
  local ocsp_responder_cert_file="${11}"
  local ocsp_issuer_cert_file="${12}"
  local ocsp_responder_cert_expected_sha="${13}"
  local ocsp_require_responder_ocsp_signing="${14}"
  local ocsp_require_responder_aki_match_ca_ski="${15}"
  local ocsp_responder_id_regex="${16}"
  python3 - "$manifest_json" "$keyring_tsv" "$expected_keyring_sha" "$expected_key_id" "$ca_file" "$crl_file" "$ocsp_response_file" "$expected_ocsp_response_sha" "$ocsp_max_age_secs" "$ocsp_require_next_update" "$ocsp_responder_cert_file" "$ocsp_issuer_cert_file" "$ocsp_responder_cert_expected_sha" "$ocsp_require_responder_ocsp_signing" "$ocsp_require_responder_aki_match_ca_ski" "$ocsp_responder_id_regex" <<'PY'
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
keyring_path = Path(sys.argv[2])
expected_keyring_sha = sys.argv[3].strip()
expected_key_id = sys.argv[4].strip()
ca_file = Path(sys.argv[5]) if sys.argv[5].strip() else None
crl_file = Path(sys.argv[6]) if sys.argv[6].strip() else None
ocsp_response_file = Path(sys.argv[7]) if sys.argv[7].strip() else None
expected_ocsp_response_sha = sys.argv[8].strip()
ocsp_max_age_secs = sys.argv[9].strip()
ocsp_require_next_update = sys.argv[10].strip() == "1"
ocsp_responder_cert_file = Path(sys.argv[11]) if sys.argv[11].strip() else None
ocsp_issuer_cert_file = Path(sys.argv[12]) if sys.argv[12].strip() else None
ocsp_responder_cert_expected_sha = sys.argv[13].strip()
ocsp_require_responder_ocsp_signing = sys.argv[14].strip() == "1"
ocsp_require_responder_aki_match_ca_ski = sys.argv[15].strip() == "1"
ocsp_responder_id_regex = sys.argv[16]

if ca_file is not None and not ca_file.is_file():
  print(
      f"refresh policy profiles manifest signer keyring CA file not readable: {ca_file}",
      file=sys.stderr,
  )
  raise SystemExit(1)
if crl_file is not None and not crl_file.is_file():
  print(
      f"refresh policy profiles manifest signer keyring CRL file not readable: {crl_file}",
      file=sys.stderr,
  )
  raise SystemExit(1)
if ocsp_response_file is not None and not ocsp_response_file.is_file():
  print(
      f"refresh policy profiles manifest signer keyring OCSP response file not readable: {ocsp_response_file}",
      file=sys.stderr,
  )
  raise SystemExit(1)
if ocsp_responder_cert_file is not None and not ocsp_responder_cert_file.is_file():
  print(
      f"refresh policy profiles manifest signer OCSP responder cert file not readable: {ocsp_responder_cert_file}",
      file=sys.stderr,
  )
  raise SystemExit(1)
if ocsp_issuer_cert_file is not None and not ocsp_issuer_cert_file.is_file():
  print(
      f"refresh policy profiles manifest signer OCSP issuer cert file not readable: {ocsp_issuer_cert_file}",
      file=sys.stderr,
  )
  raise SystemExit(1)
if ocsp_max_age_secs and not re.fullmatch(r"\d+", ocsp_max_age_secs):
  print(
      "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-max-age-secs: expected non-negative integer",
      file=sys.stderr,
  )
  raise SystemExit(1)
if expected_ocsp_response_sha and not re.fullmatch(r"[0-9a-f]{64}", expected_ocsp_response_sha):
  print(
      "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-sha256: expected 64 hex chars",
      file=sys.stderr,
  )
  raise SystemExit(1)
if ocsp_responder_cert_expected_sha and not re.fullmatch(r"[0-9a-f]{64}", ocsp_responder_cert_expected_sha):
  print(
      "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-sha256: expected 64 hex chars",
      file=sys.stderr,
  )
  raise SystemExit(1)
if crl_file is not None and ca_file is None:
  print(
      "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-crl-file requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file",
      file=sys.stderr,
  )
  raise SystemExit(1)
if ocsp_response_file is not None and ca_file is None:
  print(
      "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file",
      file=sys.stderr,
  )
  raise SystemExit(1)
if ocsp_responder_cert_file is not None and ocsp_response_file is None:
  print(
      "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-file requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file",
      file=sys.stderr,
  )
  raise SystemExit(1)
if ocsp_issuer_cert_file is not None and ocsp_response_file is None:
  print(
      "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-issuer-cert-file requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file",
      file=sys.stderr,
  )
  raise SystemExit(1)
if ocsp_responder_cert_expected_sha and ocsp_responder_cert_file is None:
  print(
      "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-sha256 requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-file",
      file=sys.stderr,
  )
  raise SystemExit(1)
if ocsp_require_responder_ocsp_signing and ocsp_responder_cert_file is None:
  print(
      "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-responder-ocsp-signing requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-file",
      file=sys.stderr,
  )
  raise SystemExit(1)
if ocsp_require_responder_aki_match_ca_ski and ocsp_responder_cert_file is None:
  print(
      "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-responder-aki-match-ca-ski requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-file",
      file=sys.stderr,
  )
  raise SystemExit(1)
if ocsp_responder_id_regex and ocsp_response_file is None:
  print(
      "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-id-regex requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file",
      file=sys.stderr,
  )
  raise SystemExit(1)

try:
  payload = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception as ex:
  print(
      f"invalid refresh policy profiles manifest: unable to parse '{manifest_path}' ({ex})",
      file=sys.stderr,
  )
  raise SystemExit(1)

if not isinstance(payload, dict):
  print("invalid refresh policy profiles manifest: expected JSON object", file=sys.stderr)
  raise SystemExit(1)

manifest_key_id = payload.get("key_id")
if not isinstance(manifest_key_id, str) or not manifest_key_id.strip():
  print(
      "invalid refresh policy profiles manifest: key_id must be non-empty string when --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv is set",
      file=sys.stderr,
  )
  raise SystemExit(1)
manifest_key_id = manifest_key_id.strip()
if expected_key_id and manifest_key_id != expected_key_id:
  print(
      f"invalid refresh policy profiles manifest: key_id mismatch (expected '{expected_key_id}', found '{manifest_key_id}')",
      file=sys.stderr,
  )
  raise SystemExit(1)

generated_at = payload.get("generated_at_utc")
generated_date = ""
if generated_at is not None:
  if not isinstance(generated_at, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", generated_at):
    print(
        "invalid refresh policy profiles manifest: generated_at_utc must be UTC RFC3339 timestamp",
        file=sys.stderr,
    )
    raise SystemExit(1)
  generated_date = generated_at[:10]

keyring_bytes = keyring_path.read_bytes()
actual_keyring_sha = hashlib.sha256(keyring_bytes).hexdigest()
if expected_keyring_sha and actual_keyring_sha != expected_keyring_sha:
  print(
      f"refresh policy profiles manifest signer keyring SHA256 mismatch: expected {expected_keyring_sha}, found {actual_keyring_sha}",
      file=sys.stderr,
  )
  raise SystemExit(1)

rows = {}

def run_openssl(command, error_prefix: str):
  result = subprocess.run(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=False,
  )
  if result.returncode != 0:
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    detail = stderr or stdout or "openssl command failed"
    print(f"{error_prefix}: {detail}", file=sys.stderr)
    raise SystemExit(1)
  return result

def parse_date(value: str, field: str) -> str:
  if not value:
    return ""
  if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
    print(
        f"invalid refresh policy profiles manifest signer keyring {field}: expected YYYY-MM-DD",
        file=sys.stderr,
    )
    raise SystemExit(1)
  try:
    datetime.strptime(value, "%Y-%m-%d")
  except ValueError:
    print(
        f"invalid refresh policy profiles manifest signer keyring {field}: invalid calendar date",
        file=sys.stderr,
    )
    raise SystemExit(1)
  return value

def parse_ocsp_time(value: str, field: str):
  raw = value.strip()
  for fmt in ("%b %d %H:%M:%S %Y GMT", "%Y-%m-%d %H:%M:%SZ"):
    try:
      return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
      continue
  print(
      f"refresh policy profiles manifest signer OCSP {field} parse failed for key_id '{manifest_key_id}': unsupported timestamp '{raw}'",
      file=sys.stderr,
  )
  raise SystemExit(1)

for line_no, raw_line in enumerate(keyring_bytes.decode("utf-8").splitlines(), start=1):
  line = raw_line.strip()
  if not line or line.startswith("#"):
    continue
  cols = raw_line.split("\t")
  if len(cols) < 2:
    print(
        f"invalid refresh policy profiles manifest signer keyring row {keyring_path}:{line_no}: expected at least key_id and public_key_file_path",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if len(cols) > 8:
    print(
        f"invalid refresh policy profiles manifest signer keyring row {keyring_path}:{line_no}: expected at most 8 columns",
        file=sys.stderr,
    )
    raise SystemExit(1)
  key_id = cols[0].strip()
  key_file_path = cols[1].strip()
  not_before = cols[2].strip() if len(cols) >= 3 else ""
  not_after = cols[3].strip() if len(cols) >= 4 else ""
  status = cols[4].strip() if len(cols) >= 5 else ""
  key_sha = cols[5].strip() if len(cols) >= 6 else ""
  cert_file_path = cols[6].strip() if len(cols) >= 7 else ""
  cert_sha = cols[7].strip() if len(cols) >= 8 else ""
  if not key_id:
    print(
        f"invalid refresh policy profiles manifest signer keyring row {keyring_path}:{line_no}: empty key_id",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if key_id in rows:
    print(
        f"invalid refresh policy profiles manifest signer keyring row {keyring_path}:{line_no}: duplicate key_id '{key_id}'",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if not key_file_path:
    print(
        f"invalid refresh policy profiles manifest signer keyring row {keyring_path}:{line_no}: empty public_key_file_path",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if status and status not in {"active", "revoked"}:
    print(
        f"invalid refresh policy profiles manifest signer keyring row {keyring_path}:{line_no}: status must be active or revoked",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if key_sha and not re.fullmatch(r"[0-9a-f]{64}", key_sha):
    print(
        f"invalid refresh policy profiles manifest signer keyring row {keyring_path}:{line_no}: key_sha256 must be 64 hex chars",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if cert_sha and not re.fullmatch(r"[0-9a-f]{64}", cert_sha):
    print(
        f"invalid refresh policy profiles manifest signer keyring row {keyring_path}:{line_no}: cert_sha256 must be 64 hex chars",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if cert_sha and not cert_file_path:
    print(
        f"invalid refresh policy profiles manifest signer keyring row {keyring_path}:{line_no}: cert_sha256 requires cert_file_path",
        file=sys.stderr,
    )
    raise SystemExit(1)
  not_before = parse_date(not_before, f"row {keyring_path}:{line_no} not_before")
  not_after = parse_date(not_after, f"row {keyring_path}:{line_no} not_after")
  if not_before and not_after and not_before > not_after:
    print(
        f"invalid refresh policy profiles manifest signer keyring row {keyring_path}:{line_no}: not_before is after not_after",
        file=sys.stderr,
    )
    raise SystemExit(1)
  rows[key_id] = (key_file_path, not_before, not_after, status, key_sha, cert_file_path, cert_sha)

if manifest_key_id not in rows:
  print(
      f"refresh policy profiles manifest signer keyring missing key_id '{manifest_key_id}' in {keyring_path}",
      file=sys.stderr,
  )
  raise SystemExit(1)

key_file_path, not_before, not_after, status, key_sha, cert_file_path, cert_sha = rows[manifest_key_id]
if status == "revoked":
  print(
      f"refresh policy profiles manifest signer key_id '{manifest_key_id}' is revoked in keyring {keyring_path}",
      file=sys.stderr,
  )
  raise SystemExit(1)

if (not_before or not_after) and not generated_date:
  print(
      f"refresh policy profiles manifest signer key_id '{manifest_key_id}' has key validity window in keyring {keyring_path} but generated_at_utc is missing",
      file=sys.stderr,
  )
  raise SystemExit(1)
if generated_date:
  if not_before and generated_date < not_before:
    print(
        f"refresh policy profiles manifest signer key_id '{manifest_key_id}' not active at generated_at_utc {generated_at} (window {not_before}..{not_after or '-'})",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if not_after and generated_date > not_after:
    print(
        f"refresh policy profiles manifest signer key_id '{manifest_key_id}' not active at generated_at_utc {generated_at} (window {not_before or '-'}..{not_after})",
        file=sys.stderr,
    )
    raise SystemExit(1)

key_path = Path(key_file_path)
if not key_path.is_absolute():
  key_path = (keyring_path.parent / key_path).resolve()
if not key_path.is_file():
  print(
      f"refresh policy profiles manifest signer public key file for key_id '{manifest_key_id}' not found: {key_path}",
      file=sys.stderr,
  )
  raise SystemExit(1)

actual_key_sha = hashlib.sha256(key_path.read_bytes()).hexdigest()
if key_sha and actual_key_sha != key_sha:
  print(
      f"refresh policy profiles manifest signer public key SHA256 mismatch for key_id '{manifest_key_id}': expected {key_sha}, found {actual_key_sha}",
      file=sys.stderr,
  )
  raise SystemExit(1)

cert_sha_resolved = ""
ca_sha_resolved = ""
crl_sha_resolved = ""
ocsp_sha_resolved = ""
ocsp_responder_cert_sha_resolved = ""
ocsp_issuer_cert_sha_resolved = ""
cert_path = None
if cert_file_path:
  cert_path = Path(cert_file_path)
  if not cert_path.is_absolute():
    cert_path = (keyring_path.parent / cert_path).resolve()
  if not cert_path.is_file():
    print(
        f"refresh policy profiles manifest signer certificate file for key_id '{manifest_key_id}' not found: {cert_path}",
        file=sys.stderr,
    )
    raise SystemExit(1)
  cert_sha_resolved = hashlib.sha256(cert_path.read_bytes()).hexdigest()
  if cert_sha and cert_sha != cert_sha_resolved:
    print(
        f"refresh policy profiles manifest signer certificate SHA256 mismatch for key_id '{manifest_key_id}': expected {cert_sha}, found {cert_sha_resolved}",
        file=sys.stderr,
    )
    raise SystemExit(1)
  with tempfile.NamedTemporaryFile(delete=False) as cert_pub_file:
    cert_pub_path = cert_pub_file.name
  try:
    cert_pub = run_openssl(
        ["openssl", "x509", "-in", str(cert_path), "-pubkey", "-noout"],
        f"refresh policy profiles manifest signer certificate public key extraction failed for key_id '{manifest_key_id}'",
    ).stdout
    Path(cert_pub_path).write_bytes(cert_pub)
    cert_der = run_openssl(
        ["openssl", "pkey", "-pubin", "-in", cert_pub_path, "-outform", "DER"],
        f"refresh policy profiles manifest signer certificate public key conversion failed for key_id '{manifest_key_id}'",
    ).stdout
    key_der = run_openssl(
        ["openssl", "pkey", "-pubin", "-in", str(key_path), "-outform", "DER"],
        f"refresh policy profiles manifest signer public key conversion failed for key_id '{manifest_key_id}'",
    ).stdout
  finally:
    Path(cert_pub_path).unlink(missing_ok=True)
  cert_pub_sha = hashlib.sha256(cert_der).hexdigest()
  key_pub_sha = hashlib.sha256(key_der).hexdigest()
  if cert_pub_sha != key_pub_sha:
    print(
        f"refresh policy profiles manifest signer certificate/public key mismatch for key_id '{manifest_key_id}'",
        file=sys.stderr,
    )
    raise SystemExit(1)

if ca_file:
  ca_sha_resolved = hashlib.sha256(ca_file.read_bytes()).hexdigest()
  if cert_path is None:
    print(
        f"refresh policy profiles manifest signer key_id '{manifest_key_id}' missing cert_file_path while --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file is set",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if crl_file is not None:
    crl_sha_resolved = hashlib.sha256(crl_file.read_bytes()).hexdigest()
    crl_next_update_raw = run_openssl(
        ["openssl", "crl", "-in", str(crl_file), "-noout", "-nextupdate", "-dateopt", "iso_8601"],
        f"refresh policy profiles manifest signer CRL nextUpdate extraction failed for key_id '{manifest_key_id}'",
    ).stdout.decode("utf-8", errors="replace").strip()
    if not crl_next_update_raw.startswith("nextUpdate="):
      print(
          f"refresh policy profiles manifest signer CRL nextUpdate parse failed for key_id '{manifest_key_id}': expected nextUpdate=... in '{crl_next_update_raw}'",
          file=sys.stderr,
      )
      raise SystemExit(1)
    crl_next_update = crl_next_update_raw.split("=", 1)[1].strip()
    try:
      crl_next_update_dt = datetime.strptime(crl_next_update, "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
      print(
          f"refresh policy profiles manifest signer CRL nextUpdate parse failed for key_id '{manifest_key_id}': unsupported timestamp '{crl_next_update}'",
          file=sys.stderr,
      )
      raise SystemExit(1)
    now_utc = datetime.now(timezone.utc)
    if crl_next_update_dt < now_utc:
      print(
          f"refresh policy profiles manifest signer CRL is stale for key_id '{manifest_key_id}': nextUpdate {crl_next_update} is before current UTC time {now_utc.strftime('%Y-%m-%d %H:%M:%SZ')}",
          file=sys.stderr,
      )
      raise SystemExit(1)
  verify_cmd = ["openssl", "verify", "-CAfile", str(ca_file)]
  if crl_file is not None:
    verify_cmd.extend(["-crl_check", "-CRLfile", str(crl_file)])
  verify_cmd.append(str(cert_path))
  run_openssl(
      verify_cmd,
      f"refresh policy profiles manifest signer certificate verification failed for key_id '{manifest_key_id}'",
  )
  if ocsp_response_file is not None:
    ocsp_issuer_cert_file_effective = ocsp_issuer_cert_file if ocsp_issuer_cert_file is not None else ca_file
    ocsp_issuer_cert_sha_resolved = hashlib.sha256(ocsp_issuer_cert_file_effective.read_bytes()).hexdigest()
    ocsp_sha_resolved = hashlib.sha256(ocsp_response_file.read_bytes()).hexdigest()
    if expected_ocsp_response_sha and ocsp_sha_resolved != expected_ocsp_response_sha:
      print(
          f"refresh policy profiles manifest signer OCSP response SHA256 mismatch for key_id '{manifest_key_id}': expected {expected_ocsp_response_sha}, found {ocsp_sha_resolved}",
          file=sys.stderr,
      )
      raise SystemExit(1)
    ocsp_verify_args = ["-VAfile", str(ca_file)]
    if ocsp_responder_cert_file is not None:
      ocsp_responder_cert_sha_resolved = hashlib.sha256(ocsp_responder_cert_file.read_bytes()).hexdigest()
      if ocsp_responder_cert_expected_sha and ocsp_responder_cert_sha_resolved != ocsp_responder_cert_expected_sha:
        print(
            f"refresh policy profiles manifest signer OCSP responder cert SHA256 mismatch for key_id '{manifest_key_id}': expected {ocsp_responder_cert_expected_sha}, found {ocsp_responder_cert_sha_resolved}",
            file=sys.stderr,
        )
        raise SystemExit(1)
      run_openssl(
          ["openssl", "verify", "-CAfile", str(ca_file), str(ocsp_responder_cert_file)],
          f"refresh policy profiles manifest signer OCSP responder cert verify failed for key_id '{manifest_key_id}'",
      )
      responder_cert_text = ""
      if ocsp_require_responder_ocsp_signing or ocsp_require_responder_aki_match_ca_ski:
        responder_cert_text = run_openssl(
            ["openssl", "x509", "-in", str(ocsp_responder_cert_file), "-noout", "-text"],
            f"refresh policy profiles manifest signer OCSP responder cert metadata extraction failed for key_id '{manifest_key_id}'",
        ).stdout.decode("utf-8", errors="replace")
      if ocsp_require_responder_ocsp_signing:
        eku_match = re.search(r"X509v3 Extended Key Usage:\s*\n((?:\s+.+\n)+)", responder_cert_text)
        eku_block = eku_match.group(1) if eku_match is not None else ""
        if "OCSP Signing" not in eku_block:
          print(
              f"refresh policy profiles manifest signer OCSP responder cert EKU missing OCSP Signing for key_id '{manifest_key_id}'",
              file=sys.stderr,
          )
          raise SystemExit(1)
      if ocsp_require_responder_aki_match_ca_ski:
        ca_cert_text = run_openssl(
            ["openssl", "x509", "-in", str(ca_file), "-noout", "-text"],
            f"refresh policy profiles manifest signer CA cert SKI extraction failed for key_id '{manifest_key_id}'",
        ).stdout.decode("utf-8", errors="replace")
        responder_aki_match = re.search(
            r"X509v3 Authority Key Identifier:\s*\n\s*(?:keyid:)?\s*([0-9A-Fa-f:]+)",
            responder_cert_text,
        )
        if responder_aki_match is None:
          print(
              f"refresh policy profiles manifest signer OCSP responder cert AKI keyid missing for key_id '{manifest_key_id}'",
              file=sys.stderr,
          )
          raise SystemExit(1)
        ca_ski_match = re.search(r"X509v3 Subject Key Identifier:\s*\n\s*([0-9A-Fa-f:]+)", ca_cert_text)
        if ca_ski_match is None:
          print(
              f"refresh policy profiles manifest signer CA cert SKI missing for key_id '{manifest_key_id}'",
              file=sys.stderr,
          )
          raise SystemExit(1)
        responder_aki = re.sub(r"[^0-9A-Fa-f]", "", responder_aki_match.group(1)).lower()
        ca_ski = re.sub(r"[^0-9A-Fa-f]", "", ca_ski_match.group(1)).lower()
        if responder_aki != ca_ski:
          print(
              f"refresh policy profiles manifest signer OCSP responder cert AKI/SKI mismatch for key_id '{manifest_key_id}': responder AKI {responder_aki} != CA SKI {ca_ski}",
              file=sys.stderr,
          )
          raise SystemExit(1)
      ocsp_verify_args = ["-verify_other", str(ocsp_responder_cert_file), "-VAfile", str(ocsp_responder_cert_file)]
    ocsp_status_raw = run_openssl(
        [
            "openssl",
            "ocsp",
            "-issuer",
            str(ocsp_issuer_cert_file_effective),
            "-cert",
            str(cert_path),
            "-respin",
            str(ocsp_response_file),
            "-CAfile",
            str(ca_file),
            *ocsp_verify_args,
            "-no_nonce",
        ],
        f"refresh policy profiles manifest signer OCSP verification failed for key_id '{manifest_key_id}'",
    ).stdout.decode("utf-8", errors="replace")
    status_match = re.search(r":\s*(good|revoked|unknown)\b", ocsp_status_raw)
    if status_match is None:
      print(
          f"refresh policy profiles manifest signer OCSP status parse failed for key_id '{manifest_key_id}'",
          file=sys.stderr,
      )
      raise SystemExit(1)
    ocsp_status = status_match.group(1)
    if ocsp_status != "good":
      print(
          f"refresh policy profiles manifest signer OCSP status is {ocsp_status} for key_id '{manifest_key_id}'",
          file=sys.stderr,
      )
      raise SystemExit(1)
    this_update_match = re.search(r"This Update:\s*(.+)", ocsp_status_raw)
    if this_update_match is None:
      print(
          f"refresh policy profiles manifest signer OCSP thisUpdate missing for key_id '{manifest_key_id}'",
          file=sys.stderr,
      )
      raise SystemExit(1)
    this_update_dt = parse_ocsp_time(this_update_match.group(1), "thisUpdate")
    now_utc = datetime.now(timezone.utc)
    if this_update_dt > now_utc + timedelta(minutes=5):
      print(
          f"refresh policy profiles manifest signer OCSP thisUpdate is in the future for key_id '{manifest_key_id}': {this_update_dt.strftime('%Y-%m-%d %H:%M:%SZ')}",
          file=sys.stderr,
      )
      raise SystemExit(1)
    if ocsp_max_age_secs:
      max_age = int(ocsp_max_age_secs)
      age_seconds = (now_utc - this_update_dt).total_seconds()
      if age_seconds > max_age:
        print(
            f"refresh policy profiles manifest signer OCSP response is stale for key_id '{manifest_key_id}': age {int(age_seconds)}s exceeds max {max_age}s",
            file=sys.stderr,
        )
        raise SystemExit(1)
    next_update_match = re.search(r"Next Update:\s*(.+)", ocsp_status_raw)
    if next_update_match is None and ocsp_require_next_update:
      print(
          f"refresh policy profiles manifest signer OCSP nextUpdate missing for key_id '{manifest_key_id}'",
          file=sys.stderr,
      )
      raise SystemExit(1)
    if next_update_match is not None:
      next_update_dt = parse_ocsp_time(next_update_match.group(1), "nextUpdate")
      if next_update_dt < now_utc:
        print(
            f"refresh policy profiles manifest signer OCSP nextUpdate is stale for key_id '{manifest_key_id}': {next_update_dt.strftime('%Y-%m-%d %H:%M:%SZ')}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if ocsp_responder_id_regex:
      ocsp_text = run_openssl(
          [
              "openssl",
              "ocsp",
              "-issuer",
              str(ocsp_issuer_cert_file_effective),
              "-cert",
              str(cert_path),
              "-respin",
              str(ocsp_response_file),
              "-CAfile",
              str(ca_file),
              *ocsp_verify_args,
              "-no_nonce",
              "-resp_text",
          ],
          f"refresh policy profiles manifest signer OCSP responder-id extraction failed for key_id '{manifest_key_id}'",
      ).stdout.decode("utf-8", errors="replace")
      responder_match = re.search(r"Responder Id:\s*(.+)", ocsp_text)
      if responder_match is None:
        print(
            f"refresh policy profiles manifest signer OCSP responder-id parse failed for key_id '{manifest_key_id}'",
            file=sys.stderr,
        )
        raise SystemExit(1)
      responder_id = responder_match.group(1).strip()
      try:
        responder_id_ok = re.search(ocsp_responder_id_regex, responder_id) is not None
      except re.error as ex:
        print(
            f"invalid --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-id-regex: {ex}",
            file=sys.stderr,
        )
        raise SystemExit(1)
      if not responder_id_ok:
        print(
            f"refresh policy profiles manifest signer OCSP responder-id mismatch for key_id '{manifest_key_id}': regex '{ocsp_responder_id_regex}' did not match '{responder_id}'",
            file=sys.stderr,
        )
        raise SystemExit(1)
elif crl_file is not None or ocsp_response_file is not None:
  print(
      "refresh policy profiles manifest signer CRL/OCSP checks require --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file",
      file=sys.stderr,
  )
  raise SystemExit(1)

print(str(key_path))
print(actual_key_sha)
print(actual_keyring_sha)
print(manifest_key_id)
print(not_before)
print(not_after)
print(cert_sha_resolved)
print(ca_sha_resolved)
print(crl_sha_resolved)
print(ocsp_sha_resolved)
print(ocsp_responder_cert_sha_resolved)
print(ocsp_issuer_cert_sha_resolved)
PY
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATE_STR="$(date +%Y-%m-%d)"
RUN_ID="${DATE_STR}-$$-$(date +%H%M%S)"
OUT_DIR=""
SV_TESTS_DIR="${HOME}/sv-tests"
VERILATOR_DIR="${HOME}/verilator-verification"
YOSYS_DIR="${HOME}/yosys/tests/sva"
OPENTITAN_DIR="${HOME}/opentitan"
AVIP_GLOB="${HOME}/mbit/*avip*"
CIRCT_VERILOG_BIN="$REPO_ROOT/build/bin/circt-verilog"
CIRCT_VERILOG_BIN_AVIP=""
CIRCT_VERILOG_BIN_OPENTITAN=""
BASELINE_FILE="utils/formal-baselines.tsv"
PLAN_FILE="PROJECT_PLAN.md"
Z3_BIN="${Z3_BIN:-}"
UPDATE_BASELINES=0
FAIL_ON_DIFF=0
STRICT_GATE=0
BASELINE_WINDOW=1
BASELINE_WINDOW_DAYS=0
FAIL_ON_NEW_XPASS=0
FAIL_ON_PASSRATE_REGRESSION=0
FAIL_ON_NEW_FAILURE_CASES=0
FAIL_ON_NEW_BMC_TIMEOUT_CASES=0
FAIL_ON_NEW_BMC_UNKNOWN_CASES=0
FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_FAIL=0
FAIL_ON_NEW_E2E_MODE_DIFF_STATUS_DIFF=0
FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_PASS=0
FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E=0
FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E_STRICT=0
declare -a FAIL_ON_NEW_OPENTITAN_LEC_STRICT_XPROP_COUNTER_KEYS=()
EXPECTED_FAILURES_FILE=""
EXPECTATIONS_DRY_RUN=0
EXPECTATIONS_DRY_RUN_REPORT_JSONL=""
EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS=5
EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE=""
EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID=""
FAIL_ON_UNEXPECTED_FAILURES=0
FAIL_ON_UNUSED_EXPECTED_FAILURES=0
PRUNE_EXPECTED_FAILURES_FILE=""
PRUNE_EXPECTED_FAILURES_DROP_UNUSED=0
REFRESH_EXPECTED_FAILURES_FILE=""
REFRESH_EXPECTED_FAILURES_INCLUDE_SUITE_REGEX=""
REFRESH_EXPECTED_FAILURES_INCLUDE_MODE_REGEX=""
EXPECTED_FAILURE_CASES_FILE=""
FAIL_ON_UNEXPECTED_FAILURE_CASES=0
FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES=0
FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES=0
PRUNE_EXPECTED_FAILURE_CASES_FILE=""
PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED=0
PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED=0
REFRESH_EXPECTED_FAILURE_CASES_FILE=""
REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON=""
REFRESH_EXPECTED_FAILURE_CASES_COLLAPSE_STATUS_ANY=0
REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_SUITE_REGEX=""
REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_MODE_REGEX=""
REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_STATUS_REGEX=""
REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_ID_REGEX=""
JSON_SUMMARY_FILE=""
LANE_STATE_TSV=""
RESUME_FROM_LANE_STATE=0
RESET_LANE_STATE=0
declare -a MERGE_LANE_STATE_TSVS=()
LANE_STATE_HMAC_KEY_FILE=""
LANE_STATE_HMAC_KEYRING_TSV=""
LANE_STATE_HMAC_KEYRING_SHA256=""
LANE_STATE_HMAC_KEY_ID=""
LANE_STATE_HMAC_EFFECTIVE_KEY_FILE=""
LANE_STATE_HMAC_KEY_NOT_BEFORE=""
LANE_STATE_HMAC_KEY_NOT_AFTER=""
LANE_STATE_HMAC_MODE="none"
LANE_STATE_HMAC_KEYRING_SHA256_RESOLVED=""
LANE_STATE_MANIFEST_SIGN_MODE="none"
LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE=""
LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE=""
LANE_STATE_MANIFEST_ED25519_KEYRING_TSV=""
LANE_STATE_MANIFEST_ED25519_KEYRING_SHA256=""
LANE_STATE_MANIFEST_ED25519_CA_FILE=""
LANE_STATE_MANIFEST_ED25519_CRL_FILE=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP=0
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY="first"
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY_SET=0
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES="file,http,https"
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET=0
LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_POLICY=""
LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_POLICY_SET=0
LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_ALLOWED_SCHEMES=""
LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET=0
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILE=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_FILE=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_FILE=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_FILE=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_MAX_AGE_SECS=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_NEXT_UPDATE=0
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_FILE=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_FILE=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_SHA256_EXPECTED=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_RESPONDER_OCSP_SIGNING=0
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_RESPONDER_AKI_MATCH_CA_SKI=0
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_ID_REGEX=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEY_ID=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_RETRIES=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_DELAY_SECS=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_TIMEOUT_SECS=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_JITTER_SECS=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_STATUS=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256=0
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN=0
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN=0
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_AGE_SECS=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS=""
LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA=0
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY="first"
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY_SET=0
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES="file,http,https"
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET=0
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_RETRIES=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_DELAY_SECS=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_TIMEOUT_SECS=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_JITTER_SECS=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_STATUS=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256=0
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN=0
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN=0
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_AGE_SECS=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS=""
LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_SHA256=""
LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE=""
LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_FILE=""
LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256_EXPECTED=""
LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_OCSP_SIGNING=0
LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_AKI_MATCH_CA_SKI=0
LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS=""
LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS_EFFECTIVE=""
LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_NEXT_UPDATE=0
LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_ID_REGEX=""
LANE_STATE_MANIFEST_ED25519_CERT_SUBJECT_REGEX=""
LANE_STATE_MANIFEST_ED25519_KEY_ID=""
LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_SHA256=""
LANE_STATE_MANIFEST_ED25519_KEYRING_SHA256_RESOLVED=""
LANE_STATE_MANIFEST_ED25519_CA_SHA256=""
LANE_STATE_MANIFEST_ED25519_CRL_SHA256=""
LANE_STATE_MANIFEST_ED25519_OCSP_SHA256=""
LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256=""
LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256_RESOLVED=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_SHA256_RESOLVED=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_MODE="none"
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_SHA256_RESOLVED=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_CERT_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_NOT_BEFORE=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_NOT_AFTER=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_EFFECTIVE_PUBLIC_KEY_FILE=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEY_ID_RESOLVED=""
LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_PROVENANCE_JSON=""
LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_PROVENANCE_JSON=""
LANE_STATE_MANIFEST_ED25519_CERT_SHA256=""
LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_NOT_BEFORE=""
LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_NOT_AFTER=""
LANE_STATE_MANIFEST_ED25519_EFFECTIVE_PUBLIC_KEY_FILE=""
LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_MODE="none"
INCLUDE_LANE_REGEX=""
EXCLUDE_LANE_REGEX=""
WITH_OPENTITAN=0
WITH_OPENTITAN_LEC_STRICT=0
WITH_OPENTITAN_E2E=0
WITH_OPENTITAN_E2E_STRICT=0
WITH_AVIP=0
OPENTITAN_LEC_IMPL_FILTER=""
OPENTITAN_LEC_INCLUDE_MASKED=0
OPENTITAN_LEC_STRICT_DUMP_UNKNOWN_SOURCES=0
OPENTITAN_E2E_SIM_TARGETS=""
OPENTITAN_E2E_VERILOG_TARGETS=""
OPENTITAN_E2E_SIM_TIMEOUT=""
OPENTITAN_E2E_IMPL_FILTER=""
OPENTITAN_E2E_INCLUDE_MASKED=0
OPENTITAN_E2E_LEC_X_MODE="xopt"
OPENTITAN_E2E_LEC_X_MODE_FLAG_COUNT=0
BMC_RUN_SMTLIB=0
BMC_ALLOW_MULTI_CLOCK=0
BMC_ASSUME_KNOWN_INPUTS=0
LEC_ASSUME_KNOWN_INPUTS=0
LEC_ACCEPT_XPROP_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --sv-tests)
      SV_TESTS_DIR="$2"; shift 2 ;;
    --verilator)
      VERILATOR_DIR="$2"; shift 2 ;;
    --yosys)
      YOSYS_DIR="$2"; shift 2 ;;
    --z3-bin)
      Z3_BIN="$2"; shift 2 ;;
    --baseline-file)
      BASELINE_FILE="$2"; shift 2 ;;
    --plan-file)
      PLAN_FILE="$2"; shift 2 ;;
    --opentitan)
      OPENTITAN_DIR="$2"; WITH_OPENTITAN=1; shift 2 ;;
    --with-opentitan)
      WITH_OPENTITAN=1; shift ;;
    --with-opentitan-lec-strict)
      WITH_OPENTITAN_LEC_STRICT=1; shift ;;
    --with-opentitan-e2e)
      WITH_OPENTITAN_E2E=1; shift ;;
    --with-opentitan-e2e-strict)
      WITH_OPENTITAN_E2E_STRICT=1; shift ;;
    --opentitan-lec-impl-filter)
      OPENTITAN_LEC_IMPL_FILTER="$2"; shift 2 ;;
    --opentitan-lec-include-masked)
      OPENTITAN_LEC_INCLUDE_MASKED=1; shift ;;
    --opentitan-lec-strict-dump-unknown-sources)
      OPENTITAN_LEC_STRICT_DUMP_UNKNOWN_SOURCES=1; shift ;;
    --opentitan-e2e-sim-targets)
      OPENTITAN_E2E_SIM_TARGETS="$2"; shift 2 ;;
    --opentitan-e2e-verilog-targets)
      OPENTITAN_E2E_VERILOG_TARGETS="$2"; shift 2 ;;
    --opentitan-e2e-sim-timeout)
      OPENTITAN_E2E_SIM_TIMEOUT="$2"; shift 2 ;;
    --opentitan-e2e-impl-filter)
      OPENTITAN_E2E_IMPL_FILTER="$2"; shift 2 ;;
    --opentitan-e2e-include-masked)
      OPENTITAN_E2E_INCLUDE_MASKED=1; shift ;;
    --opentitan-e2e-lec-x-optimistic)
      OPENTITAN_E2E_LEC_X_MODE="xopt"
      OPENTITAN_E2E_LEC_X_MODE_FLAG_COUNT=$((OPENTITAN_E2E_LEC_X_MODE_FLAG_COUNT + 1))
      shift ;;
    --opentitan-e2e-lec-strict-x)
      OPENTITAN_E2E_LEC_X_MODE="strict"
      OPENTITAN_E2E_LEC_X_MODE_FLAG_COUNT=$((OPENTITAN_E2E_LEC_X_MODE_FLAG_COUNT + 1))
      shift ;;
    --circt-verilog)
      CIRCT_VERILOG_BIN="$2"; shift 2 ;;
    --circt-verilog-avip)
      CIRCT_VERILOG_BIN_AVIP="$2"; shift 2 ;;
    --circt-verilog-opentitan)
      CIRCT_VERILOG_BIN_OPENTITAN="$2"; shift 2 ;;
    --with-avip)
      WITH_AVIP=1; shift ;;
    --avip-glob)
      AVIP_GLOB="$2"; shift 2 ;;
    --bmc-run-smtlib)
      BMC_RUN_SMTLIB=1; shift ;;
    --bmc-allow-multi-clock)
      BMC_ALLOW_MULTI_CLOCK=1; shift ;;
    --bmc-assume-known-inputs)
      BMC_ASSUME_KNOWN_INPUTS=1; shift ;;
    --lec-assume-known-inputs)
      LEC_ASSUME_KNOWN_INPUTS=1; shift ;;
    --lec-accept-xprop-only)
      LEC_ACCEPT_XPROP_ONLY=1; shift ;;
    --update-baselines)
      UPDATE_BASELINES=1; shift ;;
    --fail-on-diff)
      FAIL_ON_DIFF=1; shift ;;
    --strict-gate)
      STRICT_GATE=1; shift ;;
    --baseline-window)
      BASELINE_WINDOW="$2"; shift 2 ;;
    --baseline-window-days)
      BASELINE_WINDOW_DAYS="$2"; shift 2 ;;
    --fail-on-new-xpass)
      FAIL_ON_NEW_XPASS=1; shift ;;
    --fail-on-passrate-regression)
      FAIL_ON_PASSRATE_REGRESSION=1; shift ;;
    --fail-on-new-failure-cases)
      FAIL_ON_NEW_FAILURE_CASES=1; shift ;;
    --fail-on-new-bmc-timeout-cases)
      FAIL_ON_NEW_BMC_TIMEOUT_CASES=1; shift ;;
    --fail-on-new-bmc-unknown-cases)
      FAIL_ON_NEW_BMC_UNKNOWN_CASES=1; shift ;;
    --fail-on-new-e2e-mode-diff-strict-only-fail)
      FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_FAIL=1; shift ;;
    --fail-on-new-e2e-mode-diff-status-diff)
      FAIL_ON_NEW_E2E_MODE_DIFF_STATUS_DIFF=1; shift ;;
    --fail-on-new-e2e-mode-diff-strict-only-pass)
      FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_PASS=1; shift ;;
    --fail-on-new-e2e-mode-diff-missing-in-e2e)
      FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E=1; shift ;;
    --fail-on-new-e2e-mode-diff-missing-in-e2e-strict)
      FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E_STRICT=1; shift ;;
    --fail-on-new-opentitan-lec-strict-xprop-counter)
      FAIL_ON_NEW_OPENTITAN_LEC_STRICT_XPROP_COUNTER_KEYS+=("$2"); shift 2 ;;
    --expected-failures-file)
      EXPECTED_FAILURES_FILE="$2"; shift 2 ;;
    --expectations-dry-run)
      EXPECTATIONS_DRY_RUN=1; shift ;;
    --expectations-dry-run-report-jsonl)
      EXPECTATIONS_DRY_RUN_REPORT_JSONL="$2"; shift 2 ;;
    --expectations-dry-run-report-max-sample-rows)
      EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS="$2"; shift 2 ;;
    --expectations-dry-run-report-hmac-key-file)
      EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE="$2"; shift 2 ;;
    --expectations-dry-run-report-hmac-key-id)
      EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID="$2"; shift 2 ;;
    --fail-on-unexpected-failures)
      FAIL_ON_UNEXPECTED_FAILURES=1; shift ;;
    --fail-on-unused-expected-failures)
      FAIL_ON_UNUSED_EXPECTED_FAILURES=1; shift ;;
    --prune-expected-failures-file)
      PRUNE_EXPECTED_FAILURES_FILE="$2"; shift 2 ;;
    --prune-expected-failures-drop-unused)
      PRUNE_EXPECTED_FAILURES_DROP_UNUSED=1; shift ;;
    --refresh-expected-failures-file)
      REFRESH_EXPECTED_FAILURES_FILE="$2"; shift 2 ;;
    --refresh-expected-failures-include-suite-regex)
      REFRESH_EXPECTED_FAILURES_INCLUDE_SUITE_REGEX="$2"; shift 2 ;;
    --refresh-expected-failures-include-mode-regex)
      REFRESH_EXPECTED_FAILURES_INCLUDE_MODE_REGEX="$2"; shift 2 ;;
    --expected-failure-cases-file)
      EXPECTED_FAILURE_CASES_FILE="$2"; shift 2 ;;
    --fail-on-unexpected-failure-cases)
      FAIL_ON_UNEXPECTED_FAILURE_CASES=1; shift ;;
    --fail-on-expired-expected-failure-cases)
      FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES=1; shift ;;
    --fail-on-unmatched-expected-failure-cases)
      FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES=1; shift ;;
    --prune-expected-failure-cases-file)
      PRUNE_EXPECTED_FAILURE_CASES_FILE="$2"; shift 2 ;;
    --prune-expected-failure-cases-drop-unmatched)
      PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED=1; shift ;;
    --prune-expected-failure-cases-drop-expired)
      PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED=1; shift ;;
    --refresh-expected-failure-cases-file)
      REFRESH_EXPECTED_FAILURE_CASES_FILE="$2"; shift 2 ;;
    --refresh-expected-failure-cases-default-expires-on)
      REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON="$2"; shift 2 ;;
    --refresh-expected-failure-cases-collapse-status-any)
      REFRESH_EXPECTED_FAILURE_CASES_COLLAPSE_STATUS_ANY=1; shift ;;
    --refresh-expected-failure-cases-include-suite-regex)
      REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_SUITE_REGEX="$2"; shift 2 ;;
    --refresh-expected-failure-cases-include-mode-regex)
      REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_MODE_REGEX="$2"; shift 2 ;;
    --refresh-expected-failure-cases-include-status-regex)
      REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_STATUS_REGEX="$2"; shift 2 ;;
    --refresh-expected-failure-cases-include-id-regex)
      REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_ID_REGEX="$2"; shift 2 ;;
    --json-summary)
      JSON_SUMMARY_FILE="$2"; shift 2 ;;
    --lane-state-tsv)
      LANE_STATE_TSV="$2"; shift 2 ;;
    --resume-from-lane-state)
      RESUME_FROM_LANE_STATE=1; shift ;;
    --reset-lane-state)
      RESET_LANE_STATE=1; shift ;;
    --merge-lane-state-tsv)
      MERGE_LANE_STATE_TSVS+=("$2"); shift 2 ;;
    --lane-state-hmac-key-file)
      LANE_STATE_HMAC_KEY_FILE="$2"; shift 2 ;;
    --lane-state-hmac-keyring-tsv)
      LANE_STATE_HMAC_KEYRING_TSV="$2"; shift 2 ;;
    --lane-state-hmac-keyring-sha256)
      LANE_STATE_HMAC_KEYRING_SHA256="$2"; shift 2 ;;
    --lane-state-hmac-key-id)
      LANE_STATE_HMAC_KEY_ID="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-private-key-file)
      LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-public-key-file)
      LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-keyring-tsv)
      LANE_STATE_MANIFEST_ED25519_KEYRING_TSV="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-keyring-sha256)
      LANE_STATE_MANIFEST_ED25519_KEYRING_SHA256="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ca-file)
      LANE_STATE_MANIFEST_ED25519_CA_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-file)
      LANE_STATE_MANIFEST_ED25519_CRL_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-cmd)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-uri)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP=1; shift ;;
    --lane-state-manifest-ed25519-crl-refresh-auto-uri-policy)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY="$2"; LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY_SET=1; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-auto-uri-allowed-schemes)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES="$2"; LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET=1; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-auto-uri-policy)
      LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_POLICY="$2"; LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_POLICY_SET=1; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-auto-uri-allowed-schemes)
      LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_ALLOWED_SCHEMES="$2"; LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET=1; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-json)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profile)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-sha256)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-json)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-public-key-file)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-sha256)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_SHA256="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-crl-file)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-sha256)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_SHA256="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-max-age-secs)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_MAX_AGE_SECS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-next-update)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_NEXT_UPDATE=1; shift ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-file)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-issuer-cert-file)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-sha256)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_SHA256_EXPECTED="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-responder-ocsp-signing)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_RESPONDER_OCSP_SIGNING=1; shift ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-responder-aki-match-ca-ski)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_RESPONDER_AKI_MATCH_CA_SKI=1; shift ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-id-regex)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_ID_REGEX="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-key-id)
      LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEY_ID="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-retries)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_RETRIES="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-delay-secs)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_DELAY_SECS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-timeout-secs)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_TIMEOUT_SECS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-jitter-secs)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_JITTER_SECS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-metadata-file)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-metadata-require-transport)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-metadata-require-status)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_STATUS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-metadata-require-uri-regex)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-metadata-require-tls-peer-sha256)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-metadata-require-cert-chain-sha256)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-metadata-require-artifact-sha256)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256=1; shift ;;
    --lane-state-manifest-ed25519-crl-refresh-metadata-require-ca-cert-in-cert-chain)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN=1; shift ;;
    --lane-state-manifest-ed25519-crl-refresh-metadata-require-tls-peer-in-cert-chain)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN=1; shift ;;
    --lane-state-manifest-ed25519-crl-refresh-metadata-require-cert-chain-length-min)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-metadata-max-age-secs)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_AGE_SECS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-crl-refresh-metadata-max-future-skew-secs)
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-response-file)
      LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-cmd)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-uri)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA=1; shift ;;
    --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-policy)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY="$2"; LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY_SET=1; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-allowed-schemes)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES="$2"; LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET=1; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-retries)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_RETRIES="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-delay-secs)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_DELAY_SECS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-timeout-secs)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_TIMEOUT_SECS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-jitter-secs)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_JITTER_SECS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-metadata-file)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-transport)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-status)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_STATUS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-uri-regex)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-tls-peer-sha256)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-cert-chain-sha256)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-artifact-sha256)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256=1; shift ;;
    --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-ca-cert-in-cert-chain)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN=1; shift ;;
    --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-tls-peer-in-cert-chain)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN=1; shift ;;
    --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-cert-chain-length-min)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-metadata-max-age-secs)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_AGE_SECS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-refresh-metadata-max-future-skew-secs)
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-response-sha256)
      LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_SHA256="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-responder-cert-file)
      LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-issuer-cert-file)
      LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_FILE="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-responder-cert-sha256)
      LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256_EXPECTED="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-require-responder-ocsp-signing)
      LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_OCSP_SIGNING=1; shift ;;
    --lane-state-manifest-ed25519-ocsp-require-responder-aki-match-ca-ski)
      LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_AKI_MATCH_CA_SKI=1; shift ;;
    --lane-state-manifest-ed25519-ocsp-max-age-secs)
      LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-ocsp-require-next-update)
      LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_NEXT_UPDATE=1; shift ;;
    --lane-state-manifest-ed25519-ocsp-responder-id-regex)
      LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_ID_REGEX="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-cert-subject-regex)
      LANE_STATE_MANIFEST_ED25519_CERT_SUBJECT_REGEX="$2"; shift 2 ;;
    --lane-state-manifest-ed25519-key-id)
      LANE_STATE_MANIFEST_ED25519_KEY_ID="$2"; shift 2 ;;
    --include-lane-regex)
      INCLUDE_LANE_REGEX="$2"; shift 2 ;;
    --exclude-lane-regex)
      EXCLUDE_LANE_REGEX="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$PWD/formal-results-${DATE_STR//-/}"
fi
if ! [[ "$BASELINE_WINDOW" =~ ^[0-9]+$ ]] || [[ "$BASELINE_WINDOW" == "0" ]]; then
  echo "invalid --baseline-window: expected positive integer" >&2
  exit 1
fi
if ! [[ "$BASELINE_WINDOW_DAYS" =~ ^[0-9]+$ ]]; then
  echo "invalid --baseline-window-days: expected non-negative integer" >&2
  exit 1
fi
if ! [[ "$EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS" =~ ^[0-9]+$ ]]; then
  echo "invalid --expectations-dry-run-report-max-sample-rows: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$INCLUDE_LANE_REGEX" ]]; then
  set +e
  printf '' | grep -Eq "$INCLUDE_LANE_REGEX" 2>/dev/null
  lane_regex_ec=$?
  set -e
  if [[ "$lane_regex_ec" == "2" ]]; then
    echo "invalid --include-lane-regex: $INCLUDE_LANE_REGEX" >&2
    exit 1
  fi
fi
if [[ -n "$EXCLUDE_LANE_REGEX" ]]; then
  set +e
  printf '' | grep -Eq "$EXCLUDE_LANE_REGEX" 2>/dev/null
  lane_regex_ec=$?
  set -e
  if [[ "$lane_regex_ec" == "2" ]]; then
    echo "invalid --exclude-lane-regex: $EXCLUDE_LANE_REGEX" >&2
    exit 1
  fi
fi
if [[ "$RESUME_FROM_LANE_STATE" == "1" && -z "$LANE_STATE_TSV" ]]; then
  echo "--resume-from-lane-state requires --lane-state-tsv" >&2
  exit 1
fi
if [[ "$RESET_LANE_STATE" == "1" && -z "$LANE_STATE_TSV" ]]; then
  echo "--reset-lane-state requires --lane-state-tsv" >&2
  exit 1
fi
if [[ "${#MERGE_LANE_STATE_TSVS[@]}" -gt 0 && -z "$LANE_STATE_TSV" ]]; then
  echo "--merge-lane-state-tsv requires --lane-state-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_HMAC_KEY_FILE" && -z "$LANE_STATE_TSV" ]]; then
  echo "--lane-state-hmac-key-file requires --lane-state-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_HMAC_KEYRING_TSV" && -z "$LANE_STATE_TSV" ]]; then
  echo "--lane-state-hmac-keyring-tsv requires --lane-state-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE" && -z "$LANE_STATE_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-private-key-file requires --lane-state-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE" && -z "$LANE_STATE_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-public-key-file requires --lane-state-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" && -z "$LANE_STATE_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-keyring-tsv requires --lane-state-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_HMAC_KEY_FILE" && -n "$LANE_STATE_HMAC_KEYRING_TSV" ]]; then
  echo "--lane-state-hmac-key-file and --lane-state-hmac-keyring-tsv are mutually exclusive" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE" && -n "$LANE_STATE_HMAC_KEY_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-private-key-file is mutually exclusive with lane-state HMAC signing options" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE" && -n "$LANE_STATE_HMAC_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-private-key-file is mutually exclusive with lane-state HMAC signing options" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE" && -n "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-public-key-file and --lane-state-manifest-ed25519-keyring-tsv are mutually exclusive" >&2
  exit 1
fi
if [[ "$RESUME_FROM_LANE_STATE" == "1" && "$RESET_LANE_STATE" == "1" ]]; then
  echo "--resume-from-lane-state and --reset-lane-state are mutually exclusive" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_POLICY_SET" == "1" && ! "$LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_POLICY" =~ ^(first|last|require_single)$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-refresh-auto-uri-policy: expected one of first,last,require_single" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" ]]; then
  LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_ALLOWED_SCHEMES="$(normalize_auto_uri_allowed_schemes "$LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_ALLOWED_SCHEMES" "--lane-state-manifest-ed25519-refresh-auto-uri-allowed-schemes")"
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY_SET" == "1" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY" =~ ^(first|last|require_single)$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-auto-uri-policy: expected one of first,last,require_single" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY_SET" == "1" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY" =~ ^(first|last|require_single)$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-policy: expected one of first,last,require_single" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" ]]; then
  LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES="$(normalize_auto_uri_allowed_schemes "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES" "--lane-state-manifest-ed25519-crl-refresh-auto-uri-allowed-schemes")"
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" ]]; then
  LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES="$(normalize_auto_uri_allowed_schemes "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES" "--lane-state-manifest-ed25519-ocsp-refresh-auto-uri-allowed-schemes")"
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILE" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-json requires --lane-state-manifest-ed25519-refresh-policy-profile" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILE" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profile requires --lane-state-manifest-ed25519-refresh-policy-profiles-json" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-sha256 requires --lane-state-manifest-ed25519-refresh-policy-profiles-json" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256" && ! "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-sha256: expected 64 hex chars" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-json requires --lane-state-manifest-ed25519-refresh-policy-profiles-json" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-public-key-file requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-json" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-json" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_SHA256" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-sha256 requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_SHA256" && ! "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-sha256: expected 64 hex chars" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-crl-file requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-crl-file requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ca-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_SHA256" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-sha256 requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_SHA256" && ! "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-sha256: expected 64 hex chars" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_MAX_AGE_SECS" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-max-age-secs requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_MAX_AGE_SECS" && ! "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_MAX_AGE_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-max-age-secs: expected non-negative integer" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_NEXT_UPDATE" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-next-update requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-file requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-issuer-cert-file requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_SHA256_EXPECTED" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-sha256 requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_SHA256_EXPECTED" && ! "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_SHA256_EXPECTED" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-sha256: expected 64 hex chars" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_RESPONDER_OCSP_SIGNING" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-responder-ocsp-signing requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_RESPONDER_AKI_MATCH_CA_SKI" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-require-responder-aki-match-ca-ski requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-cert-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_ID_REGEX" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-responder-id-regex requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-ocsp-response-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_FILE" && -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-public-key-file and --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv are mutually exclusive" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEY_ID" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-key-id requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-json" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON" && ! -r "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON" ]]; then
  echo "refresh policy profiles JSON not readable: $LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON" ]]; then
  LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256_RESOLVED="$(
    sha256sum "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON" | awk '{print $1}'
  )"
  if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256" && "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256_RESOLVED" != "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256" ]]; then
    echo "refresh policy profiles SHA256 mismatch: expected $LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256, found $LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256_RESOLVED" >&2
    exit 1
  fi
else
  LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256_RESOLVED=""
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-refresh-policy-profiles-manifest-json requires --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-public-key-file or --lane-state-manifest-ed25519-refresh-policy-profiles-manifest-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON" && ! -r "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON" ]]; then
  echo "refresh policy profiles manifest JSON not readable: $LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_FILE" && ! -r "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_FILE" ]]; then
  echo "refresh policy profiles manifest public key not readable: $LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_FILE" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV" && ! -r "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV" ]]; then
  echo "refresh policy profiles manifest signer keyring TSV not readable: $LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_FILE" && ! -r "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_FILE" ]]; then
  echo "refresh policy profiles manifest signer keyring CA file not readable: $LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_FILE" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_FILE" && ! -r "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_FILE" ]]; then
  echo "refresh policy profiles manifest signer keyring CRL file not readable: $LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_FILE" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE" && ! -r "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE" ]]; then
  echo "refresh policy profiles manifest signer keyring OCSP response file not readable: $LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_FILE" && ! -r "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_FILE" ]]; then
  echo "refresh policy profiles manifest signer OCSP responder cert file not readable: $LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_FILE" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_FILE" && ! -r "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_FILE" ]]; then
  echo "refresh policy profiles manifest signer OCSP issuer cert file not readable: $LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_FILE" >&2
  exit 1
fi
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_SHA256_RESOLVED=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_MODE="none"
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_SHA256_RESOLVED=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_CERT_SHA256=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_NOT_BEFORE=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_NOT_AFTER=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_EFFECTIVE_PUBLIC_KEY_FILE=""
LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEY_ID_RESOLVED=""
if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON" ]]; then
  LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_SHA256_RESOLVED="$(
    sha256sum "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON" | awk '{print $1}'
  )"
  if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV" ]]; then
    mapfile -t refresh_profiles_manifest_keyring_resolved < <(
      resolve_refresh_policy_profiles_manifest_public_key_from_keyring \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_TSV" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_SHA256" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEY_ID" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_FILE" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_FILE" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_SHA256" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_MAX_AGE_SECS" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_NEXT_UPDATE" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_FILE" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_FILE" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_SHA256_EXPECTED" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_RESPONDER_OCSP_SIGNING" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_RESPONDER_AKI_MATCH_CA_SKI" \
        "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_ID_REGEX"
    ) || exit 1
    if [[ "${#refresh_profiles_manifest_keyring_resolved[@]}" -ne 12 ]]; then
      echo "internal error: failed to resolve refresh policy profiles manifest signer keyring" >&2
      exit 1
    fi
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_EFFECTIVE_PUBLIC_KEY_FILE="${refresh_profiles_manifest_keyring_resolved[0]}"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_SHA256="${refresh_profiles_manifest_keyring_resolved[1]}"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_SHA256_RESOLVED="${refresh_profiles_manifest_keyring_resolved[2]}"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEY_ID_RESOLVED="${refresh_profiles_manifest_keyring_resolved[3]}"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_NOT_BEFORE="${refresh_profiles_manifest_keyring_resolved[4]}"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_NOT_AFTER="${refresh_profiles_manifest_keyring_resolved[5]}"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_CERT_SHA256="${refresh_profiles_manifest_keyring_resolved[6]}"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_SHA256="${refresh_profiles_manifest_keyring_resolved[7]}"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_SHA256="${refresh_profiles_manifest_keyring_resolved[8]}"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_SHA256="${refresh_profiles_manifest_keyring_resolved[9]}"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_SHA256="${refresh_profiles_manifest_keyring_resolved[10]}"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_SHA256="${refresh_profiles_manifest_keyring_resolved[11]}"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_MODE="keyring"
  else
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_EFFECTIVE_PUBLIC_KEY_FILE="$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_FILE"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_SHA256="$(
      sha256sum "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_FILE" | awk '{print $1}'
    )"
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_CERT_SHA256=""
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_SHA256=""
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_SHA256=""
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_SHA256=""
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_SHA256=""
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_SHA256=""
    LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_MODE="file"
  fi
  profile_manifest_rows="$(
    verify_refresh_policy_profiles_manifest \
      "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_JSON" \
      "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256_RESOLVED" \
      "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_EFFECTIVE_PUBLIC_KEY_FILE" \
      "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEY_ID"
  )" || exit 1
  if [[ -n "$profile_manifest_rows" ]]; then
    while IFS=$'\t' read -r manifest_field manifest_value; do
      case "$manifest_field" in
        manifest_key_id)
          LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEY_ID_RESOLVED="$manifest_value"
          ;;
        *)
          echo "internal error: unknown refresh policy profiles manifest field '$manifest_field'" >&2
          exit 1
          ;;
      esac
    done <<< "$profile_manifest_rows"
  fi
fi

PROFILE_SHARED_AUTO_URI_POLICY=""
PROFILE_SHARED_AUTO_URI_POLICY_SET=0
PROFILE_SHARED_AUTO_URI_ALLOWED_SCHEMES=""
PROFILE_SHARED_AUTO_URI_ALLOWED_SCHEMES_SET=0
PROFILE_SHARED_REFRESH_RETRIES=""
PROFILE_SHARED_REFRESH_RETRIES_SET=0
PROFILE_SHARED_REFRESH_DELAY_SECS=""
PROFILE_SHARED_REFRESH_DELAY_SECS_SET=0
PROFILE_SHARED_REFRESH_TIMEOUT_SECS=""
PROFILE_SHARED_REFRESH_TIMEOUT_SECS_SET=0
PROFILE_SHARED_REFRESH_JITTER_SECS=""
PROFILE_SHARED_REFRESH_JITTER_SECS_SET=0
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TRANSPORT=""
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TRANSPORT_SET=0
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_STATUS=""
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_STATUS_SET=0
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_URI_REGEX=""
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_URI_REGEX_SET=0
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256=""
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256_SET=0
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256=""
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256_SET=0
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256=0
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256_SET=0
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN=""
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN_SET=0
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN=0
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN_SET=0
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN=0
PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN_SET=0
PROFILE_SHARED_REFRESH_METADATA_MAX_AGE_SECS=""
PROFILE_SHARED_REFRESH_METADATA_MAX_AGE_SECS_SET=0
PROFILE_SHARED_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS=""
PROFILE_SHARED_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS_SET=0
PROFILE_CRL_AUTO_URI_POLICY=""
PROFILE_CRL_AUTO_URI_POLICY_SET=0
PROFILE_CRL_AUTO_URI_ALLOWED_SCHEMES=""
PROFILE_CRL_AUTO_URI_ALLOWED_SCHEMES_SET=0
PROFILE_CRL_REFRESH_RETRIES=""
PROFILE_CRL_REFRESH_RETRIES_SET=0
PROFILE_CRL_REFRESH_DELAY_SECS=""
PROFILE_CRL_REFRESH_DELAY_SECS_SET=0
PROFILE_CRL_REFRESH_TIMEOUT_SECS=""
PROFILE_CRL_REFRESH_TIMEOUT_SECS_SET=0
PROFILE_CRL_REFRESH_JITTER_SECS=""
PROFILE_CRL_REFRESH_JITTER_SECS_SET=0
PROFILE_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT=""
PROFILE_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT_SET=0
PROFILE_CRL_REFRESH_METADATA_REQUIRE_STATUS=""
PROFILE_CRL_REFRESH_METADATA_REQUIRE_STATUS_SET=0
PROFILE_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX=""
PROFILE_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX_SET=0
PROFILE_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256=""
PROFILE_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256_SET=0
PROFILE_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256=""
PROFILE_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256_SET=0
PROFILE_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256=0
PROFILE_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256_SET=0
PROFILE_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN=""
PROFILE_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN_SET=0
PROFILE_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN=0
PROFILE_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN_SET=0
PROFILE_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN=0
PROFILE_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN_SET=0
PROFILE_CRL_REFRESH_METADATA_MAX_AGE_SECS=""
PROFILE_CRL_REFRESH_METADATA_MAX_AGE_SECS_SET=0
PROFILE_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS=""
PROFILE_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS_SET=0
PROFILE_OCSP_AUTO_URI_POLICY=""
PROFILE_OCSP_AUTO_URI_POLICY_SET=0
PROFILE_OCSP_AUTO_URI_ALLOWED_SCHEMES=""
PROFILE_OCSP_AUTO_URI_ALLOWED_SCHEMES_SET=0
PROFILE_OCSP_REFRESH_RETRIES=""
PROFILE_OCSP_REFRESH_RETRIES_SET=0
PROFILE_OCSP_REFRESH_DELAY_SECS=""
PROFILE_OCSP_REFRESH_DELAY_SECS_SET=0
PROFILE_OCSP_REFRESH_TIMEOUT_SECS=""
PROFILE_OCSP_REFRESH_TIMEOUT_SECS_SET=0
PROFILE_OCSP_REFRESH_JITTER_SECS=""
PROFILE_OCSP_REFRESH_JITTER_SECS_SET=0
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT=""
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT_SET=0
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_STATUS=""
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_STATUS_SET=0
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX=""
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX_SET=0
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256=""
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256_SET=0
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256=""
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256_SET=0
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256=0
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256_SET=0
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN=""
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN_SET=0
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN=0
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN_SET=0
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN=0
PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN_SET=0
PROFILE_OCSP_REFRESH_METADATA_MAX_AGE_SECS=""
PROFILE_OCSP_REFRESH_METADATA_MAX_AGE_SECS_SET=0
PROFILE_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS=""
PROFILE_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS_SET=0

if [[ -n "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON" ]]; then
  profile_policy_rows="$(
    parse_refresh_auto_uri_policy_profile \
      "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_JSON" \
      "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILE"
  )" || exit 1
  if [[ -n "$profile_policy_rows" ]]; then
    while IFS=$'\t' read -r profile_key profile_value; do
      case "$profile_key" in
        shared_auto_uri_policy)
          PROFILE_SHARED_AUTO_URI_POLICY="$profile_value"
          PROFILE_SHARED_AUTO_URI_POLICY_SET=1
          ;;
        shared_auto_uri_allowed_schemes)
          PROFILE_SHARED_AUTO_URI_ALLOWED_SCHEMES="$profile_value"
          PROFILE_SHARED_AUTO_URI_ALLOWED_SCHEMES_SET=1
          ;;
        shared_refresh_retries)
          PROFILE_SHARED_REFRESH_RETRIES="$profile_value"
          PROFILE_SHARED_REFRESH_RETRIES_SET=1
          ;;
        shared_refresh_delay_secs)
          PROFILE_SHARED_REFRESH_DELAY_SECS="$profile_value"
          PROFILE_SHARED_REFRESH_DELAY_SECS_SET=1
          ;;
        shared_refresh_timeout_secs)
          PROFILE_SHARED_REFRESH_TIMEOUT_SECS="$profile_value"
          PROFILE_SHARED_REFRESH_TIMEOUT_SECS_SET=1
          ;;
        shared_refresh_jitter_secs)
          PROFILE_SHARED_REFRESH_JITTER_SECS="$profile_value"
          PROFILE_SHARED_REFRESH_JITTER_SECS_SET=1
          ;;
        shared_refresh_metadata_require_transport)
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TRANSPORT="$profile_value"
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TRANSPORT_SET=1
          ;;
        shared_refresh_metadata_require_status)
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_STATUS="$profile_value"
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_STATUS_SET=1
          ;;
        shared_refresh_metadata_require_uri_regex)
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_URI_REGEX="$profile_value"
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_URI_REGEX_SET=1
          ;;
        shared_refresh_metadata_require_tls_peer_sha256)
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256="$profile_value"
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256_SET=1
          ;;
        shared_refresh_metadata_require_cert_chain_sha256)
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256="$profile_value"
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256_SET=1
          ;;
        shared_refresh_metadata_require_artifact_sha256)
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256="$profile_value"
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256_SET=1
          ;;
        shared_refresh_metadata_require_cert_chain_length_min)
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN="$profile_value"
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN_SET=1
          ;;
        shared_refresh_metadata_require_ca_cert_in_cert_chain)
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN="$profile_value"
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN_SET=1
          ;;
        shared_refresh_metadata_require_tls_peer_in_cert_chain)
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN="$profile_value"
          PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN_SET=1
          ;;
        shared_refresh_metadata_max_age_secs)
          PROFILE_SHARED_REFRESH_METADATA_MAX_AGE_SECS="$profile_value"
          PROFILE_SHARED_REFRESH_METADATA_MAX_AGE_SECS_SET=1
          ;;
        shared_refresh_metadata_max_future_skew_secs)
          PROFILE_SHARED_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS="$profile_value"
          PROFILE_SHARED_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS_SET=1
          ;;
        crl_auto_uri_policy)
          PROFILE_CRL_AUTO_URI_POLICY="$profile_value"
          PROFILE_CRL_AUTO_URI_POLICY_SET=1
          ;;
        crl_auto_uri_allowed_schemes)
          PROFILE_CRL_AUTO_URI_ALLOWED_SCHEMES="$profile_value"
          PROFILE_CRL_AUTO_URI_ALLOWED_SCHEMES_SET=1
          ;;
        crl_refresh_retries)
          PROFILE_CRL_REFRESH_RETRIES="$profile_value"
          PROFILE_CRL_REFRESH_RETRIES_SET=1
          ;;
        crl_refresh_delay_secs)
          PROFILE_CRL_REFRESH_DELAY_SECS="$profile_value"
          PROFILE_CRL_REFRESH_DELAY_SECS_SET=1
          ;;
        crl_refresh_timeout_secs)
          PROFILE_CRL_REFRESH_TIMEOUT_SECS="$profile_value"
          PROFILE_CRL_REFRESH_TIMEOUT_SECS_SET=1
          ;;
        crl_refresh_jitter_secs)
          PROFILE_CRL_REFRESH_JITTER_SECS="$profile_value"
          PROFILE_CRL_REFRESH_JITTER_SECS_SET=1
          ;;
        crl_refresh_metadata_require_transport)
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT="$profile_value"
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT_SET=1
          ;;
        crl_refresh_metadata_require_status)
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_STATUS="$profile_value"
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_STATUS_SET=1
          ;;
        crl_refresh_metadata_require_uri_regex)
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX="$profile_value"
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX_SET=1
          ;;
        crl_refresh_metadata_require_tls_peer_sha256)
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256="$profile_value"
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256_SET=1
          ;;
        crl_refresh_metadata_require_cert_chain_sha256)
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256="$profile_value"
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256_SET=1
          ;;
        crl_refresh_metadata_require_artifact_sha256)
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256="$profile_value"
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256_SET=1
          ;;
        crl_refresh_metadata_require_cert_chain_length_min)
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN="$profile_value"
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN_SET=1
          ;;
        crl_refresh_metadata_require_ca_cert_in_cert_chain)
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN="$profile_value"
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN_SET=1
          ;;
        crl_refresh_metadata_require_tls_peer_in_cert_chain)
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN="$profile_value"
          PROFILE_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN_SET=1
          ;;
        crl_refresh_metadata_max_age_secs)
          PROFILE_CRL_REFRESH_METADATA_MAX_AGE_SECS="$profile_value"
          PROFILE_CRL_REFRESH_METADATA_MAX_AGE_SECS_SET=1
          ;;
        crl_refresh_metadata_max_future_skew_secs)
          PROFILE_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS="$profile_value"
          PROFILE_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS_SET=1
          ;;
        ocsp_auto_uri_policy)
          PROFILE_OCSP_AUTO_URI_POLICY="$profile_value"
          PROFILE_OCSP_AUTO_URI_POLICY_SET=1
          ;;
        ocsp_auto_uri_allowed_schemes)
          PROFILE_OCSP_AUTO_URI_ALLOWED_SCHEMES="$profile_value"
          PROFILE_OCSP_AUTO_URI_ALLOWED_SCHEMES_SET=1
          ;;
        ocsp_refresh_retries)
          PROFILE_OCSP_REFRESH_RETRIES="$profile_value"
          PROFILE_OCSP_REFRESH_RETRIES_SET=1
          ;;
        ocsp_refresh_delay_secs)
          PROFILE_OCSP_REFRESH_DELAY_SECS="$profile_value"
          PROFILE_OCSP_REFRESH_DELAY_SECS_SET=1
          ;;
        ocsp_refresh_timeout_secs)
          PROFILE_OCSP_REFRESH_TIMEOUT_SECS="$profile_value"
          PROFILE_OCSP_REFRESH_TIMEOUT_SECS_SET=1
          ;;
        ocsp_refresh_jitter_secs)
          PROFILE_OCSP_REFRESH_JITTER_SECS="$profile_value"
          PROFILE_OCSP_REFRESH_JITTER_SECS_SET=1
          ;;
        ocsp_refresh_metadata_require_transport)
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT="$profile_value"
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT_SET=1
          ;;
        ocsp_refresh_metadata_require_status)
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_STATUS="$profile_value"
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_STATUS_SET=1
          ;;
        ocsp_refresh_metadata_require_uri_regex)
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX="$profile_value"
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX_SET=1
          ;;
        ocsp_refresh_metadata_require_tls_peer_sha256)
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256="$profile_value"
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256_SET=1
          ;;
        ocsp_refresh_metadata_require_cert_chain_sha256)
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256="$profile_value"
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256_SET=1
          ;;
        ocsp_refresh_metadata_require_artifact_sha256)
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256="$profile_value"
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256_SET=1
          ;;
        ocsp_refresh_metadata_require_cert_chain_length_min)
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN="$profile_value"
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN_SET=1
          ;;
        ocsp_refresh_metadata_require_ca_cert_in_cert_chain)
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN="$profile_value"
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN_SET=1
          ;;
        ocsp_refresh_metadata_require_tls_peer_in_cert_chain)
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN="$profile_value"
          PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN_SET=1
          ;;
        ocsp_refresh_metadata_max_age_secs)
          PROFILE_OCSP_REFRESH_METADATA_MAX_AGE_SECS="$profile_value"
          PROFILE_OCSP_REFRESH_METADATA_MAX_AGE_SECS_SET=1
          ;;
        ocsp_refresh_metadata_max_future_skew_secs)
          PROFILE_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS="$profile_value"
          PROFILE_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS_SET=1
          ;;
        *)
          echo "internal error: unknown refresh policy profile field '$profile_key'" >&2
          exit 1
          ;;
      esac
    done <<< "$profile_policy_rows"
  fi
fi

if [[ "$PROFILE_SHARED_AUTO_URI_POLICY_SET" == "1" && ! "$PROFILE_SHARED_AUTO_URI_POLICY" =~ ^(first|last|require_single)$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json: profile shared auto_uri_policy must be one of first,last,require_single" >&2
  exit 1
fi
if [[ "$PROFILE_CRL_AUTO_URI_POLICY_SET" == "1" && ! "$PROFILE_CRL_AUTO_URI_POLICY" =~ ^(first|last|require_single)$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json: profile crl.auto_uri_policy must be one of first,last,require_single" >&2
  exit 1
fi
if [[ "$PROFILE_OCSP_AUTO_URI_POLICY_SET" == "1" && ! "$PROFILE_OCSP_AUTO_URI_POLICY" =~ ^(first|last|require_single)$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-refresh-policy-profiles-json: profile ocsp.auto_uri_policy must be one of first,last,require_single" >&2
  exit 1
fi
if [[ "$PROFILE_SHARED_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" ]]; then
  PROFILE_SHARED_AUTO_URI_ALLOWED_SCHEMES="$(normalize_auto_uri_allowed_schemes "$PROFILE_SHARED_AUTO_URI_ALLOWED_SCHEMES" "--lane-state-manifest-ed25519-refresh-policy-profiles-json (profile shared auto_uri_allowed_schemes)")"
fi
if [[ "$PROFILE_CRL_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" ]]; then
  PROFILE_CRL_AUTO_URI_ALLOWED_SCHEMES="$(normalize_auto_uri_allowed_schemes "$PROFILE_CRL_AUTO_URI_ALLOWED_SCHEMES" "--lane-state-manifest-ed25519-refresh-policy-profiles-json (profile crl.auto_uri_allowed_schemes)")"
fi
if [[ "$PROFILE_OCSP_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" ]]; then
  PROFILE_OCSP_AUTO_URI_ALLOWED_SCHEMES="$(normalize_auto_uri_allowed_schemes "$PROFILE_OCSP_AUTO_URI_ALLOWED_SCHEMES" "--lane-state-manifest-ed25519-refresh-policy-profiles-json (profile ocsp.auto_uri_allowed_schemes)")"
fi

# Effective precedence:
# specific CLI > shared CLI > specific profile > shared profile > defaults.
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY_SET" != "1" ]]; then
  if [[ "$LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_POLICY_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY="$LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_POLICY"
  elif [[ "$PROFILE_CRL_AUTO_URI_POLICY_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY="$PROFILE_CRL_AUTO_URI_POLICY"
  elif [[ "$PROFILE_SHARED_AUTO_URI_POLICY_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY="$PROFILE_SHARED_AUTO_URI_POLICY"
  fi
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY_SET" != "1" ]]; then
  if [[ "$LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_POLICY_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY="$LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_POLICY"
  elif [[ "$PROFILE_OCSP_AUTO_URI_POLICY_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY="$PROFILE_OCSP_AUTO_URI_POLICY"
  elif [[ "$PROFILE_SHARED_AUTO_URI_POLICY_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY="$PROFILE_SHARED_AUTO_URI_POLICY"
  fi
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET" != "1" ]]; then
  if [[ "$LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES="$LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_ALLOWED_SCHEMES"
  elif [[ "$PROFILE_CRL_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES="$PROFILE_CRL_AUTO_URI_ALLOWED_SCHEMES"
  elif [[ "$PROFILE_SHARED_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES="$PROFILE_SHARED_AUTO_URI_ALLOWED_SCHEMES"
  fi
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET" != "1" ]]; then
  if [[ "$LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES="$LANE_STATE_MANIFEST_ED25519_REFRESH_AUTO_URI_ALLOWED_SCHEMES"
  elif [[ "$PROFILE_OCSP_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES="$PROFILE_OCSP_AUTO_URI_ALLOWED_SCHEMES"
  elif [[ "$PROFILE_SHARED_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES="$PROFILE_SHARED_AUTO_URI_ALLOWED_SCHEMES"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_RETRIES" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_RETRIES_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_RETRIES="$PROFILE_CRL_REFRESH_RETRIES"
  elif [[ "$PROFILE_SHARED_REFRESH_RETRIES_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_RETRIES="$PROFILE_SHARED_REFRESH_RETRIES"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_RETRIES" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_RETRIES_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_RETRIES="$PROFILE_OCSP_REFRESH_RETRIES"
  elif [[ "$PROFILE_SHARED_REFRESH_RETRIES_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_RETRIES="$PROFILE_SHARED_REFRESH_RETRIES"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_DELAY_SECS" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_DELAY_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_DELAY_SECS="$PROFILE_CRL_REFRESH_DELAY_SECS"
  elif [[ "$PROFILE_SHARED_REFRESH_DELAY_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_DELAY_SECS="$PROFILE_SHARED_REFRESH_DELAY_SECS"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_DELAY_SECS" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_DELAY_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_DELAY_SECS="$PROFILE_OCSP_REFRESH_DELAY_SECS"
  elif [[ "$PROFILE_SHARED_REFRESH_DELAY_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_DELAY_SECS="$PROFILE_SHARED_REFRESH_DELAY_SECS"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_TIMEOUT_SECS" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_TIMEOUT_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_TIMEOUT_SECS="$PROFILE_CRL_REFRESH_TIMEOUT_SECS"
  elif [[ "$PROFILE_SHARED_REFRESH_TIMEOUT_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_TIMEOUT_SECS="$PROFILE_SHARED_REFRESH_TIMEOUT_SECS"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_TIMEOUT_SECS" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_TIMEOUT_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_TIMEOUT_SECS="$PROFILE_OCSP_REFRESH_TIMEOUT_SECS"
  elif [[ "$PROFILE_SHARED_REFRESH_TIMEOUT_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_TIMEOUT_SECS="$PROFILE_SHARED_REFRESH_TIMEOUT_SECS"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_JITTER_SECS" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_JITTER_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_JITTER_SECS="$PROFILE_CRL_REFRESH_JITTER_SECS"
  elif [[ "$PROFILE_SHARED_REFRESH_JITTER_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_JITTER_SECS="$PROFILE_SHARED_REFRESH_JITTER_SECS"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_JITTER_SECS" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_JITTER_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_JITTER_SECS="$PROFILE_OCSP_REFRESH_JITTER_SECS"
  elif [[ "$PROFILE_SHARED_REFRESH_JITTER_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_JITTER_SECS="$PROFILE_SHARED_REFRESH_JITTER_SECS"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT="$PROFILE_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TRANSPORT_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TRANSPORT"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT="$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TRANSPORT_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TRANSPORT"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_STATUS" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_METADATA_REQUIRE_STATUS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_STATUS="$PROFILE_CRL_REFRESH_METADATA_REQUIRE_STATUS"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_STATUS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_STATUS="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_STATUS"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_STATUS" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_STATUS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_STATUS="$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_STATUS"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_STATUS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_STATUS="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_STATUS"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX="$PROFILE_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_URI_REGEX_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_URI_REGEX"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX="$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_URI_REGEX_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_URI_REGEX"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256="$PROFILE_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256="$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256="$PROFILE_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256="$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256"
  fi
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256" != "1" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256="$PROFILE_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256"
  fi
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256" != "1" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256="$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN="$PROFILE_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN="$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN"
  fi
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN" != "1" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN="$PROFILE_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN"
  fi
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN" != "1" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN="$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN"
  fi
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN" != "1" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN="$PROFILE_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN"
  fi
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN" != "1" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN="$PROFILE_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN="$PROFILE_SHARED_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_AGE_SECS" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_METADATA_MAX_AGE_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_AGE_SECS="$PROFILE_CRL_REFRESH_METADATA_MAX_AGE_SECS"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_MAX_AGE_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_AGE_SECS="$PROFILE_SHARED_REFRESH_METADATA_MAX_AGE_SECS"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_AGE_SECS" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_METADATA_MAX_AGE_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_AGE_SECS="$PROFILE_OCSP_REFRESH_METADATA_MAX_AGE_SECS"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_MAX_AGE_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_AGE_SECS="$PROFILE_SHARED_REFRESH_METADATA_MAX_AGE_SECS"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS" ]]; then
  if [[ "$PROFILE_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS="$PROFILE_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS="$PROFILE_SHARED_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS"
  fi
fi
if [[ -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS" ]]; then
  if [[ "$PROFILE_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS="$PROFILE_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS"
  elif [[ "$PROFILE_SHARED_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS_SET" == "1" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS="$PROFILE_SHARED_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS"
  fi
fi
if [[ -n "$LANE_STATE_TSV" && -e "$LANE_STATE_TSV" && ! -r "$LANE_STATE_TSV" ]]; then
  echo "lane state file not readable: $LANE_STATE_TSV" >&2
  exit 1
fi
if [[ "$RESUME_FROM_LANE_STATE" == "1" && ! -f "$LANE_STATE_TSV" ]]; then
  echo "lane state file not found for resume: $LANE_STATE_TSV" >&2
  exit 1
fi
for merge_lane_state_file in "${MERGE_LANE_STATE_TSVS[@]}"; do
  if [[ ! -r "$merge_lane_state_file" ]]; then
    echo "merge lane state file not readable: $merge_lane_state_file" >&2
    exit 1
  fi
done
if [[ -n "$LANE_STATE_HMAC_KEY_FILE" && ! -r "$LANE_STATE_HMAC_KEY_FILE" ]]; then
  echo "lane state HMAC key file not readable: $LANE_STATE_HMAC_KEY_FILE" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_HMAC_KEYRING_TSV" && ! -r "$LANE_STATE_HMAC_KEYRING_TSV" ]]; then
  echo "lane state HMAC keyring TSV not readable: $LANE_STATE_HMAC_KEYRING_TSV" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_HMAC_KEYRING_SHA256" && -z "$LANE_STATE_HMAC_KEYRING_TSV" ]]; then
  echo "--lane-state-hmac-keyring-sha256 requires --lane-state-hmac-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_HMAC_KEYRING_SHA256" && ! "$LANE_STATE_HMAC_KEYRING_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid --lane-state-hmac-keyring-sha256: expected 64 hex chars" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_KEYRING_SHA256" && -z "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-keyring-sha256 requires --lane-state-manifest-ed25519-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_KEYRING_SHA256" && ! "$LANE_STATE_MANIFEST_ED25519_KEYRING_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-keyring-sha256: expected 64 hex chars" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CA_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-ca-file requires --lane-state-manifest-ed25519-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-crl-file requires --lane-state-manifest-ed25519-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_CA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-file requires --lane-state-manifest-ed25519-ca-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-cmd requires --lane-state-manifest-ed25519-crl-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-uri requires --lane-state-manifest-ed25519-crl-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY_SET" == "1" && "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-auto-uri-policy requires --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" && "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-auto-uri-allowed-schemes requires --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp" >&2
  exit 1
fi
if [[ ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY" =~ ^(first|last|require_single)$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-auto-uri-policy: expected one of first,last,require_single" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp requires --lane-state-manifest-ed25519-crl-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp requires --lane-state-manifest-ed25519-keyring-tsv" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_KEY_ID" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp requires --lane-state-manifest-ed25519-key-id" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD" && -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI" ]] || \
   [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD" && "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" == "1" ]] || \
   [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI" && "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" == "1" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-cmd, --lane-state-manifest-ed25519-crl-refresh-uri, and --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp are mutually exclusive" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-uri requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_RETRIES" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI" && "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-retries requires --lane-state-manifest-ed25519-crl-refresh-cmd, --lane-state-manifest-ed25519-crl-refresh-uri, or --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_DELAY_SECS" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI" && "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-delay-secs requires --lane-state-manifest-ed25519-crl-refresh-cmd, --lane-state-manifest-ed25519-crl-refresh-uri, or --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_TIMEOUT_SECS" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI" && "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-timeout-secs requires --lane-state-manifest-ed25519-crl-refresh-cmd, --lane-state-manifest-ed25519-crl-refresh-uri, or --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_JITTER_SECS" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI" && "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-jitter-secs requires --lane-state-manifest-ed25519-crl-refresh-cmd, --lane-state-manifest-ed25519-crl-refresh-uri, or --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI" && "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-metadata-file requires --lane-state-manifest-ed25519-crl-refresh-cmd, --lane-state-manifest-ed25519-crl-refresh-uri, or --lane-state-manifest-ed25519-crl-refresh-auto-uri-from-cert-cdp" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI" =~ ^(file|http|https)://.+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-uri: expected URI with file://, http://, or https:// scheme" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-metadata-require-transport requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_STATUS" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-metadata-require-status requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-metadata-require-uri-regex requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-metadata-require-tls-peer-sha256 requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-metadata-require-cert-chain-sha256 requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-metadata-require-artifact-sha256 requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-metadata-require-ca-cert-in-cert-chain requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-metadata-require-tls-peer-in-cert-chain requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-metadata-require-cert-chain-length-min requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_AGE_SECS" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-metadata-max-age-secs requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS" && -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-crl-refresh-metadata-max-future-skew-secs requires --lane-state-manifest-ed25519-crl-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_RETRIES" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_RETRIES" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-retries: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_DELAY_SECS" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_DELAY_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-delay-secs: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_TIMEOUT_SECS" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_TIMEOUT_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-timeout-secs: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_JITTER_SECS" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_JITTER_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-jitter-secs: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT" =~ ^(file|http|https|other)$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-metadata-require-transport: expected one of file,http,https,other" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_STATUS" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_STATUS" =~ ^(ok|error|stale|unknown)$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-metadata-require-status: expected one of ok,error,stale,unknown" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX" ]]; then
  set +e
  printf '' | grep -Eq "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX" 2>/dev/null
  crl_metadata_uri_regex_ec=$?
  set -e
  if [[ "$crl_metadata_uri_regex_ec" == "2" ]]; then
    echo "invalid --lane-state-manifest-ed25519-crl-refresh-metadata-require-uri-regex: $LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX" >&2
    exit 1
  fi
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-metadata-require-tls-peer-sha256: expected 64 lowercase hex chars" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-metadata-require-cert-chain-sha256: expected 64 lowercase hex chars" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-metadata-require-cert-chain-length-min: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_AGE_SECS" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_AGE_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-metadata-max-age-secs: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS" && ! "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-crl-refresh-metadata-max-future-skew-secs: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-response-file requires --lane-state-manifest-ed25519-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_CA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-response-file requires --lane-state-manifest-ed25519-ca-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-cmd requires --lane-state-manifest-ed25519-ocsp-response-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-uri requires --lane-state-manifest-ed25519-ocsp-response-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY_SET" == "1" && "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-auto-uri-policy requires --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES_SET" == "1" && "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-auto-uri-allowed-schemes requires --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia" >&2
  exit 1
fi
if [[ ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY" =~ ^(first|last|require_single)$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-policy: expected one of first,last,require_single" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia requires --lane-state-manifest-ed25519-ocsp-response-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia requires --lane-state-manifest-ed25519-keyring-tsv" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_KEY_ID" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia requires --lane-state-manifest-ed25519-key-id" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD" && -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI" ]] || \
   [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD" && "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" == "1" ]] || \
   [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI" && "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" == "1" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-cmd, --lane-state-manifest-ed25519-ocsp-refresh-uri, and --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia are mutually exclusive" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-uri requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_RETRIES" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI" && "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-retries requires --lane-state-manifest-ed25519-ocsp-refresh-cmd, --lane-state-manifest-ed25519-ocsp-refresh-uri, or --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_DELAY_SECS" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI" && "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-delay-secs requires --lane-state-manifest-ed25519-ocsp-refresh-cmd, --lane-state-manifest-ed25519-ocsp-refresh-uri, or --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_TIMEOUT_SECS" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI" && "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-timeout-secs requires --lane-state-manifest-ed25519-ocsp-refresh-cmd, --lane-state-manifest-ed25519-ocsp-refresh-uri, or --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_JITTER_SECS" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI" && "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-jitter-secs requires --lane-state-manifest-ed25519-ocsp-refresh-cmd, --lane-state-manifest-ed25519-ocsp-refresh-uri, or --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI" && "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" != "1" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-metadata-file requires --lane-state-manifest-ed25519-ocsp-refresh-cmd, --lane-state-manifest-ed25519-ocsp-refresh-uri, or --lane-state-manifest-ed25519-ocsp-refresh-auto-uri-from-cert-aia" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI" =~ ^(file|http|https)://.+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-uri: expected URI with file://, http://, or https:// scheme" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-transport requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_STATUS" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-status requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-uri-regex requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-tls-peer-sha256 requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-cert-chain-sha256 requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-artifact-sha256 requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-ca-cert-in-cert-chain requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-tls-peer-in-cert-chain requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-metadata-require-cert-chain-length-min requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_AGE_SECS" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-metadata-max-age-secs requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-refresh-metadata-max-future-skew-secs requires --lane-state-manifest-ed25519-ocsp-refresh-metadata-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_RETRIES" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_RETRIES" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-retries: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_DELAY_SECS" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_DELAY_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-delay-secs: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_TIMEOUT_SECS" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_TIMEOUT_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-timeout-secs: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_JITTER_SECS" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_JITTER_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-jitter-secs: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT" =~ ^(file|http|https|other)$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-transport: expected one of file,http,https,other" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_STATUS" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_STATUS" =~ ^(ok|error|stale|unknown)$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-status: expected one of ok,error,stale,unknown" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX" ]]; then
  set +e
  printf '' | grep -Eq "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX" 2>/dev/null
  ocsp_metadata_uri_regex_ec=$?
  set -e
  if [[ "$ocsp_metadata_uri_regex_ec" == "2" ]]; then
    echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-uri-regex: $LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX" >&2
    exit 1
  fi
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-tls-peer-sha256: expected 64 lowercase hex chars" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-cert-chain-sha256: expected 64 lowercase hex chars" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-metadata-require-cert-chain-length-min: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_AGE_SECS" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_AGE_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-metadata-max-age-secs: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-refresh-metadata-max-future-skew-secs: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_SHA256" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-response-sha256 requires --lane-state-manifest-ed25519-ocsp-response-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_SHA256" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-response-sha256: expected 64 hex chars" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-responder-cert-file requires --lane-state-manifest-ed25519-ocsp-response-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-issuer-cert-file requires --lane-state-manifest-ed25519-ocsp-response-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256_EXPECTED" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-responder-cert-sha256 requires --lane-state-manifest-ed25519-ocsp-responder-cert-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256_EXPECTED" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256_EXPECTED" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-responder-cert-sha256: expected 64 hex chars" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_OCSP_SIGNING" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-require-responder-ocsp-signing requires --lane-state-manifest-ed25519-ocsp-responder-cert-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_AKI_MATCH_CA_SKI" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-require-responder-aki-match-ca-ski requires --lane-state-manifest-ed25519-ocsp-responder-cert-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-max-age-secs requires --lane-state-manifest-ed25519-ocsp-response-file" >&2
  exit 1
fi
if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_NEXT_UPDATE" == "1" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-require-next-update requires --lane-state-manifest-ed25519-ocsp-response-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_ID_REGEX" && -z "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-ocsp-responder-id-regex requires --lane-state-manifest-ed25519-ocsp-response-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS" && ! "$LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS" =~ ^[0-9]+$ ]]; then
  echo "invalid --lane-state-manifest-ed25519-ocsp-max-age-secs: expected non-negative integer" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CERT_SUBJECT_REGEX" && -z "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-cert-subject-regex requires --lane-state-manifest-ed25519-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_HMAC_KEY_ID" && -z "$LANE_STATE_HMAC_KEY_FILE" && -z "$LANE_STATE_HMAC_KEYRING_TSV" ]]; then
  echo "--lane-state-hmac-key-id requires --lane-state-hmac-key-file or --lane-state-hmac-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_HMAC_KEYRING_TSV" && -z "$LANE_STATE_HMAC_KEY_ID" ]]; then
  echo "--lane-state-hmac-keyring-tsv requires --lane-state-hmac-key-id" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE" && \
      -z "$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE" && \
      -z "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" ]]; then
  echo "--lane-state-manifest-ed25519-private-key-file requires --lane-state-manifest-ed25519-public-key-file or --lane-state-manifest-ed25519-keyring-tsv" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE" && -z "$LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-public-key-file requires --lane-state-manifest-ed25519-private-key-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" && -z "$LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE" ]]; then
  echo "--lane-state-manifest-ed25519-keyring-tsv requires --lane-state-manifest-ed25519-private-key-file" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" && -z "$LANE_STATE_MANIFEST_ED25519_KEY_ID" ]]; then
  echo "--lane-state-manifest-ed25519-keyring-tsv requires --lane-state-manifest-ed25519-key-id" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" && ! -r "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" ]]; then
  echo "lane state Ed25519 keyring TSV not readable: $LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CA_FILE" && ! -r "$LANE_STATE_MANIFEST_ED25519_CA_FILE" ]]; then
  echo "lane state Ed25519 CA file not readable: $LANE_STATE_MANIFEST_ED25519_CA_FILE" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_FILE" && \
      -z "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD" && \
      ! -r "$LANE_STATE_MANIFEST_ED25519_CRL_FILE" ]]; then
  echo "lane state Ed25519 CRL file not readable: $LANE_STATE_MANIFEST_ED25519_CRL_FILE" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" && \
      -z "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD" && \
      ! -r "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" ]]; then
  echo "lane state Ed25519 OCSP response file not readable: $LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE" && ! -r "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE" ]]; then
  echo "lane state Ed25519 OCSP responder cert file not readable: $LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_FILE" && ! -r "$LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_FILE" ]]; then
  echo "lane state Ed25519 OCSP issuer cert file not readable: $LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_FILE" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" ]]; then
  if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS" ]]; then
    LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS_EFFECTIVE="$LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS"
  else
    LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS_EFFECTIVE="604800"
  fi
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE" && ! -r "$LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE" ]]; then
  echo "lane state Ed25519 private key file not readable: $LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE" >&2
  exit 1
fi
if [[ -n "$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE" && ! -r "$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE" ]]; then
  echo "lane state Ed25519 public key file not readable: $LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE" >&2
  exit 1
fi
if [[ -n "$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE" && \
      ! -r "$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE" ]]; then
  echo "expectations dry-run report HMAC key file not readable: $EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE" >&2
  exit 1
fi
if [[ -n "$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID" && \
      -z "$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE" ]]; then
  echo "--expectations-dry-run-report-hmac-key-id requires --expectations-dry-run-report-hmac-key-file" >&2
  exit 1
fi
if [[ "$FAIL_ON_UNEXPECTED_FAILURES" == "1" && -z "$EXPECTED_FAILURES_FILE" ]]; then
  echo "--fail-on-unexpected-failures requires --expected-failures-file" >&2
  exit 1
fi
if [[ "$FAIL_ON_UNUSED_EXPECTED_FAILURES" == "1" && -z "$EXPECTED_FAILURES_FILE" ]]; then
  echo "--fail-on-unused-expected-failures requires --expected-failures-file" >&2
  exit 1
fi
if [[ -n "$PRUNE_EXPECTED_FAILURES_FILE" && -z "$EXPECTED_FAILURES_FILE" ]]; then
  EXPECTED_FAILURES_FILE="$PRUNE_EXPECTED_FAILURES_FILE"
fi
if [[ -n "$PRUNE_EXPECTED_FAILURES_FILE" && "$PRUNE_EXPECTED_FAILURES_DROP_UNUSED" != "1" ]]; then
  PRUNE_EXPECTED_FAILURES_DROP_UNUSED=1
fi
if [[ -n "$PRUNE_EXPECTED_FAILURES_FILE" && ! -r "$PRUNE_EXPECTED_FAILURES_FILE" ]]; then
  echo "prune expected-failures file not readable: $PRUNE_EXPECTED_FAILURES_FILE" >&2
  exit 1
fi
if [[ -n "$EXPECTED_FAILURES_FILE" && ! -r "$EXPECTED_FAILURES_FILE" ]]; then
  echo "expected-failures file not readable: $EXPECTED_FAILURES_FILE" >&2
  exit 1
fi
if [[ "$FAIL_ON_UNEXPECTED_FAILURE_CASES" == "1" && -z "$EXPECTED_FAILURE_CASES_FILE" ]]; then
  echo "--fail-on-unexpected-failure-cases requires --expected-failure-cases-file" >&2
  exit 1
fi
if [[ "$FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES" == "1" && -z "$EXPECTED_FAILURE_CASES_FILE" ]]; then
  echo "--fail-on-expired-expected-failure-cases requires --expected-failure-cases-file" >&2
  exit 1
fi
if [[ "$FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES" == "1" && -z "$EXPECTED_FAILURE_CASES_FILE" ]]; then
  echo "--fail-on-unmatched-expected-failure-cases requires --expected-failure-cases-file" >&2
  exit 1
fi
if [[ -n "$PRUNE_EXPECTED_FAILURE_CASES_FILE" && -z "$EXPECTED_FAILURE_CASES_FILE" ]]; then
  EXPECTED_FAILURE_CASES_FILE="$PRUNE_EXPECTED_FAILURE_CASES_FILE"
fi
if [[ -n "$PRUNE_EXPECTED_FAILURE_CASES_FILE" && \
      "$PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED" != "1" && \
      "$PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED" != "1" ]]; then
  PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED=1
  PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED=1
fi
if [[ -n "$PRUNE_EXPECTED_FAILURE_CASES_FILE" && ! -r "$PRUNE_EXPECTED_FAILURE_CASES_FILE" ]]; then
  echo "prune expected-failure-cases file not readable: $PRUNE_EXPECTED_FAILURE_CASES_FILE" >&2
  exit 1
fi
if [[ -n "$EXPECTED_FAILURE_CASES_FILE" && ! -r "$EXPECTED_FAILURE_CASES_FILE" ]]; then
  echo "expected-failure-cases file not readable: $EXPECTED_FAILURE_CASES_FILE" >&2
  exit 1
fi
if [[ -n "$REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON" ]]; then
  if ! [[ "$REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "invalid --refresh-expected-failure-cases-default-expires-on: expected YYYY-MM-DD" >&2
    exit 1
  fi
fi

if [[ -z "$CIRCT_VERILOG_BIN_AVIP" ]]; then
  CIRCT_VERILOG_BIN_AVIP="$CIRCT_VERILOG_BIN"
fi
if [[ -z "$CIRCT_VERILOG_BIN_OPENTITAN" ]]; then
  CIRCT_VERILOG_BIN_OPENTITAN="$CIRCT_VERILOG_BIN"
fi

if [[ "$WITH_OPENTITAN" == "1" || \
      "$WITH_OPENTITAN_LEC_STRICT" == "1" || \
      "$WITH_OPENTITAN_E2E" == "1" || \
      "$WITH_OPENTITAN_E2E_STRICT" == "1" ]]; then
  if [[ ! -x "$CIRCT_VERILOG_BIN_OPENTITAN" ]]; then
    echo "circt-verilog for OpenTitan not executable: $CIRCT_VERILOG_BIN_OPENTITAN" >&2
    exit 1
  fi
fi
if [[ "$WITH_AVIP" == "1" ]]; then
  if [[ ! -x "$CIRCT_VERILOG_BIN_AVIP" ]]; then
    echo "circt-verilog for AVIP not executable: $CIRCT_VERILOG_BIN_AVIP" >&2
    exit 1
  fi
fi
if [[ "$STRICT_GATE" == "1" ]]; then
  FAIL_ON_NEW_XPASS=1
  FAIL_ON_PASSRATE_REGRESSION=1
  FAIL_ON_NEW_FAILURE_CASES=1
  FAIL_ON_NEW_BMC_TIMEOUT_CASES=1
  FAIL_ON_NEW_BMC_UNKNOWN_CASES=1
  FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_FAIL=1
  FAIL_ON_NEW_E2E_MODE_DIFF_STATUS_DIFF=1
  FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_PASS=1
  FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E=1
  FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E_STRICT=1
fi
for xprop_key in "${FAIL_ON_NEW_OPENTITAN_LEC_STRICT_XPROP_COUNTER_KEYS[@]}"; do
  if [[ ! "$xprop_key" =~ ^[a-z][a-z0-9_]*$ ]]; then
    echo "invalid --fail-on-new-opentitan-lec-strict-xprop-counter: expected [a-z][a-z0-9_]*, got '$xprop_key'" >&2
    exit 1
  fi
done
OPENTITAN_LEC_STRICT_XPROP_COUNTER_KEYS_CSV=""
if [[ "${#FAIL_ON_NEW_OPENTITAN_LEC_STRICT_XPROP_COUNTER_KEYS[@]}" -gt 0 ]]; then
  OPENTITAN_LEC_STRICT_XPROP_COUNTER_KEYS_CSV="$(IFS=,; echo "${FAIL_ON_NEW_OPENTITAN_LEC_STRICT_XPROP_COUNTER_KEYS[*]}")"
fi
if [[ "$OPENTITAN_E2E_LEC_X_MODE_FLAG_COUNT" -gt 1 ]]; then
  echo "Use only one of --opentitan-e2e-lec-x-optimistic or --opentitan-e2e-lec-strict-x." >&2
  exit 1
fi
if [[ -z "$JSON_SUMMARY_FILE" ]]; then
  JSON_SUMMARY_FILE="$OUT_DIR/summary.json"
fi

lane_state_ed25519_build_refresh_provenance_json() {
  local artifact_kind="$1"
  local max_attempts="$2"
  local attempt_count="$3"
  local outcome="$4"
  local attempts_tsv="$5"
  local source_metadata_json="$6"
  local source_metadata_sha256="$7"
  REFRESH_ARTIFACT_KIND="$artifact_kind" \
  REFRESH_MAX_ATTEMPTS="$max_attempts" \
  REFRESH_ATTEMPT_COUNT="$attempt_count" \
  REFRESH_OUTCOME="$outcome" \
  REFRESH_ATTEMPTS_TSV="$attempts_tsv" \
  REFRESH_SOURCE_METADATA_JSON="$source_metadata_json" \
  REFRESH_SOURCE_METADATA_SHA256="$source_metadata_sha256" \
  python3 - <<'PY'
import json
import os

attempt_rows = []
raw_rows = os.environ.get("REFRESH_ATTEMPTS_TSV", "")
for line in raw_rows.splitlines():
  if not line.strip():
    continue
  fields = line.split("\t")
  if len(fields) != 8:
    print("invalid refresh provenance row shape", file=os.sys.stderr)
    raise SystemExit(1)
  artifact_sha = fields[6].strip()
  source_metadata_sha = fields[7].strip()
  attempt_rows.append(
      {
          "attempt": int(fields[0]),
          "started_at_utc": fields[1],
          "ended_at_utc": fields[2],
          "exit_code": int(fields[3]),
          "timed_out": fields[4] == "1",
          "artifact_readable": fields[5] == "1",
          "artifact_sha256": artifact_sha if artifact_sha else None,
          "source_metadata_sha256": source_metadata_sha if source_metadata_sha else None,
      }
  )

payload = {
    "schema_version": 1,
    "artifact_kind": os.environ["REFRESH_ARTIFACT_KIND"],
    "max_attempts": int(os.environ["REFRESH_MAX_ATTEMPTS"]),
    "attempts_used": int(os.environ["REFRESH_ATTEMPT_COUNT"]),
    "outcome": os.environ["REFRESH_OUTCOME"],
    "attempts": attempt_rows,
}
source_metadata_sha = os.environ.get("REFRESH_SOURCE_METADATA_SHA256", "").strip()
if source_metadata_sha:
  payload["source_metadata_sha256"] = source_metadata_sha
source_metadata_json = os.environ.get("REFRESH_SOURCE_METADATA_JSON", "").strip()
if source_metadata_json:
  try:
    parsed_source_metadata = json.loads(source_metadata_json)
  except Exception as ex:
    print(f"invalid refresh source metadata payload: {ex}", file=os.sys.stderr)
    raise SystemExit(1)
  if not isinstance(parsed_source_metadata, dict):
    print("invalid refresh source metadata payload: expected JSON object", file=os.sys.stderr)
    raise SystemExit(1)
  payload["source_metadata"] = parsed_source_metadata
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
PY
}

lane_state_ed25519_read_refresh_source_metadata_json() {
  local metadata_file="$1"
  local require_transport="$2"
  local require_status="$3"
  local require_uri_regex="$4"
  local require_tls_peer_sha256="$5"
  local require_cert_chain_sha256="$6"
  local require_ca_cert_sha256="$7"
  local require_tls_peer_in_cert_chain="$8"
  local require_cert_chain_length_min="$9"
  local max_age_secs="${10}"
  local max_future_skew_secs="${11}"
  local require_artifact_sha256="${12}"
  local actual_artifact_sha256="${13}"
  python3 - "$metadata_file" "$require_transport" "$require_status" "$require_uri_regex" "$require_tls_peer_sha256" "$require_cert_chain_sha256" "$require_ca_cert_sha256" "$require_tls_peer_in_cert_chain" "$require_cert_chain_length_min" "$max_age_secs" "$max_future_skew_secs" "$require_artifact_sha256" "$actual_artifact_sha256" <<'PY'
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

metadata_path = Path(sys.argv[1])
require_transport = sys.argv[2].strip()
require_status = sys.argv[3].strip()
require_uri_regex = sys.argv[4]
require_tls_peer_sha256 = sys.argv[5].strip()
require_cert_chain_sha256 = sys.argv[6].strip()
require_ca_cert_sha256 = sys.argv[7].strip()
require_tls_peer_in_cert_chain = sys.argv[8].strip() == "1"
require_cert_chain_length_min_text = sys.argv[9].strip()
max_age_secs_text = sys.argv[10].strip()
max_future_skew_secs_text = sys.argv[11].strip()
require_artifact_sha256 = sys.argv[12].strip() == "1"
actual_artifact_sha256 = sys.argv[13].strip()
require_cert_chain_length_min = (
    int(require_cert_chain_length_min_text)
    if require_cert_chain_length_min_text
    else None
)
max_age_secs = int(max_age_secs_text) if max_age_secs_text else None
max_future_skew_secs = int(max_future_skew_secs_text) if max_future_skew_secs_text else None
try:
  parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
except Exception as ex:
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': {ex}",
      file=sys.stderr,
  )
  raise SystemExit(1)
if not isinstance(parsed, dict):
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': expected JSON object",
      file=sys.stderr,
  )
  raise SystemExit(1)

unknown_keys = sorted(
    set(parsed.keys())
    - {
        "schema_version",
        "source",
        "transport",
        "uri",
        "fetched_at_utc",
        "status",
        "http_status",
        "tls_peer_sha256",
        "cert_chain_sha256",
        "artifact_sha256",
        "error",
    }
)
if unknown_keys:
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': unknown key '{unknown_keys[0]}'",
      file=sys.stderr,
  )
  raise SystemExit(1)

schema_version = parsed.get("schema_version")
if schema_version != 1:
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': schema_version must be 1",
      file=sys.stderr,
  )
  raise SystemExit(1)

source = parsed.get("source")
if not isinstance(source, str) or not source.strip():
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': source must be non-empty string",
      file=sys.stderr,
  )
  raise SystemExit(1)

transport = parsed.get("transport")
if transport not in {"file", "http", "https", "other"}:
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': transport must be one of file,http,https,other",
      file=sys.stderr,
  )
  raise SystemExit(1)

uri = parsed.get("uri")
if not isinstance(uri, str) or not uri.strip():
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': uri must be non-empty string",
      file=sys.stderr,
  )
  raise SystemExit(1)

fetched_at_utc = parsed.get("fetched_at_utc")
if (
    not isinstance(fetched_at_utc, str)
    or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", fetched_at_utc)
):
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': fetched_at_utc must be UTC RFC3339 timestamp",
      file=sys.stderr,
  )
  raise SystemExit(1)
try:
  fetched_at_dt = datetime.strptime(fetched_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
except ValueError:
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': fetched_at_utc must be valid UTC RFC3339 timestamp",
      file=sys.stderr,
  )
  raise SystemExit(1)

status = parsed.get("status")
if status not in {"ok", "error", "stale", "unknown"}:
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': status must be one of ok,error,stale,unknown",
      file=sys.stderr,
  )
  raise SystemExit(1)

http_status = parsed.get("http_status")
if http_status is not None:
  if not isinstance(http_status, int) or http_status < 100 or http_status > 599:
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': http_status must be integer in [100,599]",
        file=sys.stderr,
    )
    raise SystemExit(1)
if transport in {"http", "https"} and http_status is None:
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': {transport} transport requires http_status",
      file=sys.stderr,
  )
  raise SystemExit(1)
if status == "ok" and http_status is not None and not (200 <= http_status < 400):
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': status=ok requires http_status in [200,399]",
      file=sys.stderr,
  )
  raise SystemExit(1)

tls_peer_sha256 = parsed.get("tls_peer_sha256")
if tls_peer_sha256 is not None:
  if not isinstance(tls_peer_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", tls_peer_sha256):
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': tls_peer_sha256 must be 64 lowercase hex chars",
        file=sys.stderr,
    )
    raise SystemExit(1)

cert_chain_sha256 = parsed.get("cert_chain_sha256")
if cert_chain_sha256 is not None:
  if not isinstance(cert_chain_sha256, list):
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': cert_chain_sha256 must be array of 64 lowercase hex strings",
        file=sys.stderr,
    )
    raise SystemExit(1)
  for idx, digest in enumerate(cert_chain_sha256):
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
      print(
          f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': cert_chain_sha256[{idx}] must be 64 lowercase hex chars",
          file=sys.stderr,
      )
      raise SystemExit(1)
if transport == "https" and tls_peer_sha256 is None:
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': https transport requires tls_peer_sha256",
      file=sys.stderr,
  )
  raise SystemExit(1)
if transport == "https":
  if cert_chain_sha256 is None or len(cert_chain_sha256) == 0:
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': https transport requires non-empty cert_chain_sha256",
      file=sys.stderr,
    )
    raise SystemExit(1)

artifact_sha256 = parsed.get("artifact_sha256")
if artifact_sha256 is not None:
  if not isinstance(artifact_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256):
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': artifact_sha256 must be 64 lowercase hex chars",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if actual_artifact_sha256 and artifact_sha256 != actual_artifact_sha256:
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': artifact_sha256 mismatch (expected '{actual_artifact_sha256}', got '{artifact_sha256}')",
        file=sys.stderr,
    )
    raise SystemExit(1)

if require_artifact_sha256:
  if artifact_sha256 is None:
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': artifact_sha256 required by policy",
        file=sys.stderr,
    )
    raise SystemExit(1)

error_text = parsed.get("error")
if error_text is not None and not isinstance(error_text, str):
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': error must be string",
      file=sys.stderr,
  )
  raise SystemExit(1)

if require_transport and transport != require_transport:
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': transport policy mismatch (expected '{require_transport}', got '{transport}')",
      file=sys.stderr,
  )
  raise SystemExit(1)

if require_status and status != require_status:
  print(
      f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': status policy mismatch (expected '{require_status}', got '{status}')",
      file=sys.stderr,
  )
  raise SystemExit(1)

if require_uri_regex:
  try:
    uri_regex = re.compile(require_uri_regex)
  except re.error as ex:
    print(
        f"invalid lane state Ed25519 refresh metadata policy regex '{require_uri_regex}': {ex}",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if not uri_regex.search(uri):
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': uri policy mismatch (regex '{require_uri_regex}', got '{uri}')",
        file=sys.stderr,
    )
    raise SystemExit(1)

if require_tls_peer_sha256:
  if tls_peer_sha256 != require_tls_peer_sha256:
    observed = tls_peer_sha256 if tls_peer_sha256 is not None else "<missing>"
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': tls_peer_sha256 policy mismatch (expected '{require_tls_peer_sha256}', got '{observed}')",
        file=sys.stderr,
    )
    raise SystemExit(1)

if require_cert_chain_sha256:
  if cert_chain_sha256 is None or require_cert_chain_sha256 not in cert_chain_sha256:
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': cert_chain_sha256 policy mismatch (missing '{require_cert_chain_sha256}')",
        file=sys.stderr,
    )
    raise SystemExit(1)

if require_ca_cert_sha256:
  if cert_chain_sha256 is None or require_ca_cert_sha256 not in cert_chain_sha256:
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': ca_cert_sha256 policy mismatch (missing '{require_ca_cert_sha256}')",
        file=sys.stderr,
    )
    raise SystemExit(1)

if require_tls_peer_in_cert_chain:
  if transport != "https":
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': tls_peer_sha256-in-cert_chain policy mismatch (requires https transport, got '{transport}')",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if cert_chain_sha256 is None or tls_peer_sha256 not in cert_chain_sha256:
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': tls_peer_sha256-in-cert_chain policy mismatch (tls_peer_sha256 not present in cert_chain_sha256)",
        file=sys.stderr,
    )
    raise SystemExit(1)

if require_cert_chain_length_min is not None:
  observed_len = len(cert_chain_sha256) if cert_chain_sha256 is not None else 0
  if observed_len < require_cert_chain_length_min:
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': cert_chain_sha256 length policy mismatch (min={require_cert_chain_length_min}, got={observed_len})",
        file=sys.stderr,
    )
    raise SystemExit(1)

if max_age_secs is not None:
  max_age_delta = timedelta(seconds=max_age_secs)
  age_delta = datetime.now(timezone.utc) - fetched_at_dt
  if age_delta > max_age_delta:
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': fetched_at_utc age policy mismatch (age={int(age_delta.total_seconds())}s exceeds max_age={max_age_secs}s)",
        file=sys.stderr,
    )
    raise SystemExit(1)

if max_future_skew_secs is not None:
  max_future_delta = timedelta(seconds=max_future_skew_secs)
  future_delta = fetched_at_dt - datetime.now(timezone.utc)
  if future_delta > max_future_delta:
    print(
        f"invalid lane state Ed25519 refresh metadata file '{metadata_path}': fetched_at_utc future-skew policy mismatch (future_skew={int(future_delta.total_seconds())}s exceeds max_future_skew={max_future_skew_secs}s)",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(json.dumps(parsed, sort_keys=True, separators=(",", ":")))
PY
}

lane_state_ed25519_fetch_refresh_uri_artifact() {
  local refresh_uri="$1"
  local artifact_file="$2"
  local refresh_metadata_file="$3"
  local timeout_secs="$4"
  python3 - "$refresh_uri" "$artifact_file" "$refresh_metadata_file" "$timeout_secs" <<'PY'
import hashlib
import json
import re
import socket
import ssl
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

refresh_uri = sys.argv[1]
artifact_path = Path(sys.argv[2])
metadata_path_text = sys.argv[3].strip()
timeout_secs_text = sys.argv[4].strip()
timeout_secs = int(timeout_secs_text) if timeout_secs_text else 0
timeout_value = timeout_secs if timeout_secs > 0 else None

metadata_path = Path(metadata_path_text) if metadata_path_text else None
parsed_uri = urllib.parse.urlparse(refresh_uri)
scheme = parsed_uri.scheme.lower()
if scheme not in {"file", "http", "https"}:
  print(f"unsupported refresh URI scheme '{scheme}'", file=sys.stderr)
  raise SystemExit(1)

metadata = {
    "schema_version": 1,
    "source": "built_in_fetch",
    "transport": scheme,
    "uri": refresh_uri,
    "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "unknown",
}
if metadata_path is not None:
  metadata_path.parent.mkdir(parents=True, exist_ok=True)

def write_metadata() -> None:
  if metadata_path is not None:
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

def write_artifact(data: bytes) -> None:
  artifact_path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      mode="wb",
      delete=False,
      dir=str(artifact_path.parent),
      prefix=f"{artifact_path.name}.tmp.",
  ) as tmp_handle:
    tmp_handle.write(data)
    tmp_name = tmp_handle.name
  Path(tmp_name).replace(artifact_path)

def extract_cert_chain_sha256_from_pem_blob(pem_blob: str):
  cert_hashes = []
  for match in re.findall(
      r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
      pem_blob,
      flags=re.DOTALL,
  ):
    try:
      cert_der = ssl.PEM_cert_to_DER_cert(match)
    except ValueError:
      continue
    cert_hashes.append(hashlib.sha256(cert_der).hexdigest())
  return cert_hashes

def fetch_cert_chain_sha256_via_openssl(host: str, port: int):
  if not host:
    return []
  command = [
      "openssl",
      "s_client",
      "-connect",
      f"{host}:{port}",
      "-showcerts",
      "-servername",
      host,
  ]
  try:
    completed = subprocess.run(
        command,
        input=b"",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_value,
    )
  except (OSError, subprocess.TimeoutExpired):
    return []
  pem_blob = completed.stdout.decode("utf-8", errors="replace")
  cert_hashes = extract_cert_chain_sha256_from_pem_blob(pem_blob)
  return cert_hashes

try:
  if scheme == "file":
    local_path = Path(urllib.request.url2pathname(parsed_uri.path))
    if not str(local_path):
      raise ValueError("file URI must include path")
    payload = local_path.read_bytes()
    metadata["artifact_sha256"] = hashlib.sha256(payload).hexdigest()
    write_artifact(payload)
    metadata["status"] = "ok"
  else:
    request = urllib.request.Request(refresh_uri, headers={"User-Agent": "run_formal_all.sh"})
    with urllib.request.urlopen(request, timeout=timeout_value) as response:
      payload = response.read()
      status = int(response.getcode() or 0)
      metadata["http_status"] = status
      if scheme == "https":
        cert_chain_sha256 = []
        raw = getattr(response, "fp", None)
        raw = getattr(raw, "raw", None)
        sock = getattr(raw, "_sock", None)
        if isinstance(sock, ssl.SSLSocket):
          cert_der = sock.getpeercert(binary_form=True)
          if cert_der:
            cert_chain_sha256 = [hashlib.sha256(cert_der).hexdigest()]
        if not cert_chain_sha256:
          host = parsed_uri.hostname
          port = parsed_uri.port or 443
          if host:
            cert_pem = ssl.get_server_certificate((host, port))
            cert_der = ssl.PEM_cert_to_DER_cert(cert_pem)
            cert_chain_sha256 = [hashlib.sha256(cert_der).hexdigest()]
        openssl_chain = fetch_cert_chain_sha256_via_openssl(
            parsed_uri.hostname or "",
            parsed_uri.port or 443,
        )
        if openssl_chain:
          if cert_chain_sha256:
            leaf_sha = cert_chain_sha256[0]
            if openssl_chain[0] != leaf_sha and leaf_sha not in openssl_chain:
              openssl_chain.insert(0, leaf_sha)
          cert_chain_sha256 = openssl_chain
        if cert_chain_sha256:
          metadata["tls_peer_sha256"] = cert_chain_sha256[0]
          metadata["cert_chain_sha256"] = cert_chain_sha256
      metadata["artifact_sha256"] = hashlib.sha256(payload).hexdigest()
      write_artifact(payload)
      metadata["status"] = "ok" if 200 <= status < 400 else "error"
except urllib.error.HTTPError as ex:
  metadata["status"] = "error"
  metadata["http_status"] = int(ex.code)
  metadata["error"] = str(ex)
  write_metadata()
  raise SystemExit(124 if isinstance(ex.reason, socket.timeout) else 1)
except (urllib.error.URLError, TimeoutError, socket.timeout) as ex:
  metadata["status"] = "error"
  metadata["error"] = str(ex)
  write_metadata()
  timed_out = isinstance(ex, (TimeoutError, socket.timeout))
  if isinstance(ex, urllib.error.URLError):
    timed_out = isinstance(ex.reason, socket.timeout)
  raise SystemExit(124 if timed_out else 1)
except Exception as ex:
  metadata["status"] = "error"
  metadata["error"] = str(ex)
  write_metadata()
  raise SystemExit(1)

write_metadata()
PY
}

lane_state_ed25519_resolve_refresh_uri_from_cert_extension() {
  local keyring_tsv="$1"
  local key_id="$2"
  local selector="$3"
  local selection_policy="$4"
  local allowed_schemes_csv="$5"
  python3 - "$keyring_tsv" "$key_id" "$selector" "$selection_policy" "$allowed_schemes_csv" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

keyring_path = Path(sys.argv[1])
target_key_id = sys.argv[2].strip()
selector = sys.argv[3].strip()
selection_policy = sys.argv[4].strip()
allowed_schemes_csv = sys.argv[5].strip()

if selector not in {"crl_cdp", "ocsp_aia"}:
  print(f"invalid selector '{selector}'", file=sys.stderr)
  raise SystemExit(1)
if selection_policy not in {"first", "last", "require_single"}:
  print(f"invalid selection policy '{selection_policy}'", file=sys.stderr)
  raise SystemExit(1)
allowed_schemes = set()
for token in allowed_schemes_csv.split(","):
  tok = token.strip().lower()
  if tok not in {"file", "http", "https"}:
    print(
        f"invalid allowed schemes '{allowed_schemes_csv}': expected comma-separated subset of file,http,https",
        file=sys.stderr,
    )
    raise SystemExit(1)
  allowed_schemes.add(tok)
if not allowed_schemes:
  print(
      f"invalid allowed schemes '{allowed_schemes_csv}': expected comma-separated subset of file,http,https",
      file=sys.stderr,
  )
  raise SystemExit(1)

if selector == "crl_cdp":
  ext_name = "crlDistributionPoints"
  ext_label = "CRL Distribution Points"
else:
  ext_name = "authorityInfoAccess"
  ext_label = "Authority Information Access"

rows = {}
for line_no, raw_line in enumerate(
    keyring_path.read_text(encoding="utf-8").splitlines(), start=1
):
  line = raw_line.strip()
  if not line or line.startswith("#"):
    continue
  cols = raw_line.split("\t")
  if len(cols) < 2:
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: expected at least key_id and public_key_file_path",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if len(cols) > 8:
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: expected at most 8 columns",
        file=sys.stderr,
    )
    raise SystemExit(1)
  key_id = cols[0].strip()
  cert_file_path = cols[6].strip() if len(cols) >= 7 else ""
  if not key_id:
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: empty key_id",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if key_id in rows:
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: duplicate key_id '{key_id}'",
        file=sys.stderr,
    )
    raise SystemExit(1)
  rows[key_id] = cert_file_path

if target_key_id not in rows:
  print(
      f"lane state Ed25519 keyring missing key_id '{target_key_id}' in {keyring_path}",
      file=sys.stderr,
  )
  raise SystemExit(1)

cert_file_path = rows[target_key_id]
if not cert_file_path:
  print(
      f"lane state Ed25519 key_id '{target_key_id}' missing cert_file_path in keyring {keyring_path}",
      file=sys.stderr,
  )
  raise SystemExit(1)

cert_path = Path(cert_file_path)
if not cert_path.is_absolute():
  cert_path = (keyring_path.parent / cert_path).resolve()
if not cert_path.is_file():
  print(
      f"lane state Ed25519 certificate file for key_id '{target_key_id}' not found: {cert_path}",
      file=sys.stderr,
  )
  raise SystemExit(1)

result = subprocess.run(
    ["openssl", "x509", "-in", str(cert_path), "-noout", "-ext", ext_name],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)
if result.returncode != 0:
  detail = (
      result.stderr.decode("utf-8", errors="replace").strip()
      or result.stdout.decode("utf-8", errors="replace").strip()
      or "openssl command failed"
  )
  print(
      f"lane state Ed25519 certificate {ext_label} extraction failed for key_id '{target_key_id}': {detail}",
      file=sys.stderr,
  )
  raise SystemExit(1)

ext_text = result.stdout.decode("utf-8", errors="replace")
if selector == "crl_cdp":
  uri_candidates = re.findall(r"URI:([^\s,]+)", ext_text)
else:
  uri_candidates = re.findall(
      r"OCSP\s*-\s*URI:([^\s,]+)",
      ext_text,
      flags=re.IGNORECASE,
  )

if not uri_candidates:
  print(
      f"lane state Ed25519 certificate {ext_label} has no usable URI for key_id '{target_key_id}'",
      file=sys.stderr,
  )
  raise SystemExit(1)

usable_uri_candidates = []
for candidate in uri_candidates:
  candidate_uri = candidate.strip()
  match = re.fullmatch(r"([a-z]+)://.+", candidate_uri)
  if not match:
    continue
  candidate_scheme = match.group(1).lower()
  if candidate_scheme not in {"file", "http", "https"}:
    continue
  if candidate_scheme not in allowed_schemes:
    continue
  usable_uri_candidates.append(candidate_uri)
if not usable_uri_candidates:
  print(
      f"lane state Ed25519 certificate {ext_label} has no URI matching allowed schemes '{allowed_schemes_csv}' for key_id '{target_key_id}'",
      file=sys.stderr,
  )
  raise SystemExit(1)
if selection_policy == "require_single" and len(usable_uri_candidates) != 1:
  print(
      f"lane state Ed25519 certificate {ext_label} has {len(usable_uri_candidates)} usable URIs for key_id '{target_key_id}' while --auto-uri-policy=require_single",
      file=sys.stderr,
  )
  raise SystemExit(1)
if selection_policy == "last":
  refresh_uri = usable_uri_candidates[-1]
else:
  refresh_uri = usable_uri_candidates[0]

print(refresh_uri)
PY
}

run_lane_state_ed25519_refresh_hook() {
  local artifact_kind="$1"
  local artifact_name="$2"
  local refresh_cmd="$3"
  local refresh_uri="$4"
  local artifact_file="$5"
  local refresh_retries="$6"
  local refresh_delay_secs="$7"
  local refresh_timeout_secs="$8"
  local refresh_jitter_secs="$9"
  local provenance_var_name="${10}"
  local refresh_metadata_file="${11}"
  local metadata_require_transport="${12}"
  local metadata_require_status="${13}"
  local metadata_require_uri_regex="${14}"
  local metadata_require_tls_peer_sha256="${15}"
  local metadata_require_cert_chain_sha256="${16}"
  local metadata_require_ca_cert_sha256="${17}"
  local metadata_require_tls_peer_in_cert_chain="${18}"
  local metadata_require_cert_chain_length_min="${19}"
  local metadata_max_age_secs="${20}"
  local metadata_max_future_skew_secs="${21}"
  local metadata_require_artifact_sha256="${22}"
  local retry_count="${refresh_retries:-0}"
  local delay_secs="${refresh_delay_secs:-0}"
  local timeout_secs="${refresh_timeout_secs:-0}"
  local jitter_secs="${refresh_jitter_secs:-0}"
  local max_attempts="$((retry_count + 1))"
  local attempt=1
  local cmd_rc=0
  local sleep_secs=0
  local attempt_started_utc=""
  local attempt_ended_utc=""
  local timed_out=0
  local artifact_readable=0
  local artifact_sha256=""
  local source_metadata_sha256=""
  local source_metadata_json=""
  local final_source_metadata_sha256=""
  local final_source_metadata_json=""
  local outcome="success"
  local attempts_tsv=""
  local attempt_count=0
  local provenance_json=""
  if [[ -z "$refresh_cmd" && -z "$refresh_uri" ]]; then
    return
  fi
  while (( attempt <= max_attempts )); do
    attempt_started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    timed_out=0
    artifact_readable=0
    artifact_sha256=""
    source_metadata_sha256=""
    source_metadata_json=""
    if [[ -n "$refresh_uri" ]]; then
      if lane_state_ed25519_fetch_refresh_uri_artifact \
        "$refresh_uri" \
        "$artifact_file" \
        "$refresh_metadata_file" \
        "$timeout_secs"; then
        cmd_rc=0
      else
        cmd_rc=$?
      fi
    elif (( timeout_secs > 0 )); then
      if LANE_STATE_MANIFEST_ED25519_CA_FILE="$LANE_STATE_MANIFEST_ED25519_CA_FILE" \
         LANE_STATE_MANIFEST_ED25519_CRL_FILE="$LANE_STATE_MANIFEST_ED25519_CRL_FILE" \
         LANE_STATE_MANIFEST_ED25519_KEYRING_TSV="$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" \
         LANE_STATE_MANIFEST_ED25519_KEY_ID="$LANE_STATE_MANIFEST_ED25519_KEY_ID" \
         LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE="$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" \
         LANE_STATE_MANIFEST_ED25519_REFRESH_METADATA_FILE="$refresh_metadata_file" \
         python3 - "$refresh_cmd" "$timeout_secs" <<'PY'
import subprocess
import sys

cmd = sys.argv[1]
timeout_secs = int(sys.argv[2])
try:
  completed = subprocess.run(
      ["bash", "-lc", cmd],
      check=False,
      timeout=timeout_secs,
  )
except subprocess.TimeoutExpired:
  raise SystemExit(124)
raise SystemExit(completed.returncode)
PY
      then
        cmd_rc=0
      else
        cmd_rc=$?
      fi
    else
      if LANE_STATE_MANIFEST_ED25519_CA_FILE="$LANE_STATE_MANIFEST_ED25519_CA_FILE" \
         LANE_STATE_MANIFEST_ED25519_CRL_FILE="$LANE_STATE_MANIFEST_ED25519_CRL_FILE" \
         LANE_STATE_MANIFEST_ED25519_KEYRING_TSV="$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" \
         LANE_STATE_MANIFEST_ED25519_KEY_ID="$LANE_STATE_MANIFEST_ED25519_KEY_ID" \
         LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE="$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" \
         LANE_STATE_MANIFEST_ED25519_REFRESH_METADATA_FILE="$refresh_metadata_file" \
         bash -lc "$refresh_cmd"; then
        cmd_rc=0
      else
        cmd_rc=$?
      fi
    fi
    attempt_ended_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ "$cmd_rc" == "124" ]]; then
      timed_out=1
    fi
    if [[ "$cmd_rc" == "0" ]]; then
      if [[ -r "$artifact_file" ]]; then
        artifact_readable=1
        artifact_sha256="$(sha256sum "$artifact_file" | awk '{print $1}')"
        if [[ -n "$refresh_metadata_file" ]]; then
          if [[ ! -r "$refresh_metadata_file" ]]; then
            outcome="metadata_unreadable"
            echo "lane state Ed25519 ${artifact_name} refresh metadata file not readable after refresh attempt ${attempt}/${max_attempts}: $refresh_metadata_file" >&2
            attempt_count="$attempt"
            attempts_tsv+="${attempt}"$'\t'"${attempt_started_utc}"$'\t'"${attempt_ended_utc}"$'\t'"${cmd_rc}"$'\t'"${timed_out}"$'\t'"${artifact_readable}"$'\t'"${artifact_sha256}"$'\t'"${source_metadata_sha256}"$'\n'
            if (( attempt >= max_attempts )); then
              provenance_json="$(lane_state_ed25519_build_refresh_provenance_json "$artifact_kind" "$max_attempts" "$attempt_count" "$outcome" "$attempts_tsv" "$final_source_metadata_json" "$final_source_metadata_sha256")"
              if [[ -n "$provenance_json" ]]; then
                printf -v "$provenance_var_name" '%s' "$provenance_json"
              fi
              exit 1
            fi
            sleep_secs="$delay_secs"
            if (( jitter_secs > 0 )); then
              sleep_secs=$((sleep_secs + (RANDOM % (jitter_secs + 1))))
            fi
            if (( sleep_secs > 0 )); then
              sleep "$sleep_secs"
            fi
            attempt=$((attempt + 1))
            continue
          fi
          source_metadata_sha256="$(sha256sum "$refresh_metadata_file" | awk '{print $1}')"
          source_metadata_json="$(
            lane_state_ed25519_read_refresh_source_metadata_json \
              "$refresh_metadata_file" \
              "$metadata_require_transport" \
              "$metadata_require_status" \
              "$metadata_require_uri_regex" \
              "$metadata_require_tls_peer_sha256" \
              "$metadata_require_cert_chain_sha256" \
              "$metadata_require_ca_cert_sha256" \
              "$metadata_require_tls_peer_in_cert_chain" \
              "$metadata_require_cert_chain_length_min" \
              "$metadata_max_age_secs" \
              "$metadata_max_future_skew_secs" \
              "$metadata_require_artifact_sha256" \
              "$artifact_sha256"
          )"
          final_source_metadata_sha256="$source_metadata_sha256"
          final_source_metadata_json="$source_metadata_json"
        fi
        attempt_count="$attempt"
        attempts_tsv+="${attempt}"$'\t'"${attempt_started_utc}"$'\t'"${attempt_ended_utc}"$'\t'"${cmd_rc}"$'\t'"${timed_out}"$'\t'"${artifact_readable}"$'\t'"${artifact_sha256}"$'\t'"${source_metadata_sha256}"$'\n'
        provenance_json="$(lane_state_ed25519_build_refresh_provenance_json "$artifact_kind" "$max_attempts" "$attempt_count" "$outcome" "$attempts_tsv" "$final_source_metadata_json" "$final_source_metadata_sha256")"
        if [[ -n "$provenance_json" ]]; then
          printf -v "$provenance_var_name" '%s' "$provenance_json"
        fi
        return
      fi
      outcome="artifact_unreadable"
      echo "lane state Ed25519 ${artifact_name} file not readable after refresh attempt ${attempt}/${max_attempts}: $artifact_file" >&2
    elif [[ "$cmd_rc" == "124" ]]; then
      outcome="timeout"
      if [[ -n "$refresh_uri" ]]; then
        echo "lane state Ed25519 ${artifact_name} refresh URI timed out on attempt ${attempt}/${max_attempts} (timeout=${timeout_secs}s): $refresh_uri" >&2
      else
        echo "lane state Ed25519 ${artifact_name} refresh command timed out on attempt ${attempt}/${max_attempts} (timeout=${timeout_secs}s): $refresh_cmd" >&2
      fi
    else
      outcome="cmd_failed"
      if [[ -n "$refresh_uri" ]]; then
        echo "lane state Ed25519 ${artifact_name} refresh URI failed on attempt ${attempt}/${max_attempts}: $refresh_uri" >&2
      else
        echo "lane state Ed25519 ${artifact_name} refresh command failed on attempt ${attempt}/${max_attempts}: $refresh_cmd" >&2
      fi
    fi
    attempt_count="$attempt"
    attempts_tsv+="${attempt}"$'\t'"${attempt_started_utc}"$'\t'"${attempt_ended_utc}"$'\t'"${cmd_rc}"$'\t'"${timed_out}"$'\t'"${artifact_readable}"$'\t'"${artifact_sha256}"$'\t'"${source_metadata_sha256}"$'\n'
    if (( attempt >= max_attempts )); then
      provenance_json="$(lane_state_ed25519_build_refresh_provenance_json "$artifact_kind" "$max_attempts" "$attempt_count" "$outcome" "$attempts_tsv" "$final_source_metadata_json" "$final_source_metadata_sha256")"
      if [[ -n "$provenance_json" ]]; then
        printf -v "$provenance_var_name" '%s' "$provenance_json"
      fi
      exit 1
    fi
    sleep_secs="$delay_secs"
    if (( jitter_secs > 0 )); then
      sleep_secs=$((sleep_secs + (RANDOM % (jitter_secs + 1))))
    fi
    if (( sleep_secs > 0 )); then
      sleep "$sleep_secs"
    fi
    attempt=$((attempt + 1))
  done
}

if [[ -n "$LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE" ]]; then
  if ! command -v openssl >/dev/null 2>&1; then
    echo "lane-state Ed25519 manifest mode requires openssl in PATH" >&2
    exit 1
  fi
  if [[ -n "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" ]]; then
    if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP" == "1" ]]; then
      LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI="$(
        lane_state_ed25519_resolve_refresh_uri_from_cert_extension \
          "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" \
          "$LANE_STATE_MANIFEST_ED25519_KEY_ID" \
          "crl_cdp" \
          "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY" \
          "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES"
      )"
    fi
    if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA" == "1" ]]; then
      LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI="$(
        lane_state_ed25519_resolve_refresh_uri_from_cert_extension \
          "$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" \
          "$LANE_STATE_MANIFEST_ED25519_KEY_ID" \
          "ocsp_aia" \
          "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY" \
          "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES"
      )"
    fi
    LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_PROVENANCE_JSON=""
    LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_PROVENANCE_JSON=""
    crl_refresh_metadata_require_ca_cert_sha256=""
    ocsp_refresh_metadata_require_ca_cert_sha256=""
    if [[ "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN" == "1" ]]; then
      crl_refresh_metadata_require_ca_cert_sha256="$(sha256sum "$LANE_STATE_MANIFEST_ED25519_CA_FILE" | awk '{print $1}')"
    fi
    if [[ "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN" == "1" ]]; then
      ocsp_refresh_metadata_require_ca_cert_sha256="$(sha256sum "$LANE_STATE_MANIFEST_ED25519_CA_FILE" | awk '{print $1}')"
    fi
    run_lane_state_ed25519_refresh_hook \
      "crl" \
      "CRL" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_FILE" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_RETRIES" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_DELAY_SECS" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_TIMEOUT_SECS" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_JITTER_SECS" \
      "LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_PROVENANCE_JSON" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_STATUS" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256" \
      "$crl_refresh_metadata_require_ca_cert_sha256" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_AGE_SECS" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS" \
      "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256"
    run_lane_state_ed25519_refresh_hook \
      "ocsp" \
      "OCSP response" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_RETRIES" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_DELAY_SECS" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_TIMEOUT_SECS" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_JITTER_SECS" \
      "LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_PROVENANCE_JSON" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_STATUS" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256" \
      "$ocsp_refresh_metadata_require_ca_cert_sha256" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_AGE_SECS" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS" \
      "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256"
    mapfile -t lane_state_ed25519_keyring_resolved < <(
      LANE_STATE_MANIFEST_ED25519_KEYRING_TSV="$LANE_STATE_MANIFEST_ED25519_KEYRING_TSV" \
      LANE_STATE_MANIFEST_ED25519_KEYRING_SHA256="$LANE_STATE_MANIFEST_ED25519_KEYRING_SHA256" \
      LANE_STATE_MANIFEST_ED25519_KEY_ID="$LANE_STATE_MANIFEST_ED25519_KEY_ID" \
      LANE_STATE_MANIFEST_ED25519_CA_FILE="$LANE_STATE_MANIFEST_ED25519_CA_FILE" \
      LANE_STATE_MANIFEST_ED25519_CRL_FILE="$LANE_STATE_MANIFEST_ED25519_CRL_FILE" \
      LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE="$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" \
      LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_SHA256="$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_SHA256" \
      LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE="$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE" \
      LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_FILE="$LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_FILE" \
      LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256_EXPECTED="$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256_EXPECTED" \
      LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_OCSP_SIGNING="$LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_OCSP_SIGNING" \
      LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_AKI_MATCH_CA_SKI="$LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_AKI_MATCH_CA_SKI" \
      LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS="$LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS_EFFECTIVE" \
      LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_NEXT_UPDATE="$LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_NEXT_UPDATE" \
      LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_ID_REGEX="$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_ID_REGEX" \
      LANE_STATE_MANIFEST_ED25519_CERT_SUBJECT_REGEX="$LANE_STATE_MANIFEST_ED25519_CERT_SUBJECT_REGEX" \
      python3 - <<'PY'
import hashlib
from datetime import datetime, timedelta, timezone
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

keyring_path = Path(os.environ["LANE_STATE_MANIFEST_ED25519_KEYRING_TSV"])
expected_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_KEYRING_SHA256", "").strip()
target_key_id = os.environ["LANE_STATE_MANIFEST_ED25519_KEY_ID"].strip()
ca_file = os.environ.get("LANE_STATE_MANIFEST_ED25519_CA_FILE", "").strip()
crl_file = os.environ.get("LANE_STATE_MANIFEST_ED25519_CRL_FILE", "").strip()
ocsp_response_file = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE", "").strip()
ocsp_response_expected_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_SHA256", "").strip()
ocsp_responder_cert_file = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE", "").strip()
ocsp_issuer_cert_file = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_FILE", "").strip()
ocsp_responder_cert_expected_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256_EXPECTED", "").strip()
ocsp_require_responder_ocsp_signing = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_OCSP_SIGNING", "").strip() == "1"
ocsp_require_responder_aki_match_ca_ski = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_AKI_MATCH_CA_SKI", "").strip() == "1"
ocsp_max_age_secs = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS", "").strip()
ocsp_require_next_update = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_NEXT_UPDATE", "").strip() == "1"
ocsp_responder_id_regex = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_ID_REGEX", "").strip()
cert_subject_regex = os.environ.get("LANE_STATE_MANIFEST_ED25519_CERT_SUBJECT_REGEX", "").strip()

keyring_bytes = keyring_path.read_bytes()
actual_sha = hashlib.sha256(keyring_bytes).hexdigest()
if expected_sha and actual_sha != expected_sha:
  print(
      f"lane state Ed25519 keyring SHA256 mismatch: expected {expected_sha}, found {actual_sha}",
      file=sys.stderr,
  )
  raise SystemExit(1)

rows = {}

def run_openssl(command, error_prefix: str):
  result = subprocess.run(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=False,
  )
  if result.returncode != 0:
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    detail = stderr or stdout or "openssl command failed"
    print(f"{error_prefix}: {detail}", file=sys.stderr)
    raise SystemExit(1)
  return result

def parse_date(value: str, field: str) -> str:
  if not value:
    return ""
  if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
    print(
        f"invalid lane state Ed25519 keyring {field}: expected YYYY-MM-DD",
        file=sys.stderr,
    )
    raise SystemExit(1)
  try:
    datetime.strptime(value, "%Y-%m-%d")
  except ValueError:
    print(
        f"invalid lane state Ed25519 keyring {field}: invalid calendar date",
        file=sys.stderr,
    )
    raise SystemExit(1)
  return value

def parse_ocsp_time(value: str, field: str):
  raw = value.strip()
  for fmt in ("%b %d %H:%M:%S %Y GMT", "%Y-%m-%d %H:%M:%SZ"):
    try:
      return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
      continue
  print(
      f"lane state Ed25519 OCSP {field} parse failed for key_id '{target_key_id}': unsupported timestamp '{raw}'",
      file=sys.stderr,
  )
  raise SystemExit(1)

for line_no, raw_line in enumerate(keyring_bytes.decode("utf-8").splitlines(), start=1):
  line = raw_line.strip()
  if not line or line.startswith("#"):
    continue
  cols = raw_line.split("\t")
  if len(cols) < 2:
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: expected at least key_id and public_key_file_path",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if len(cols) > 8:
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: expected at most 8 columns",
        file=sys.stderr,
    )
    raise SystemExit(1)
  key_id = cols[0].strip()
  key_file_path = cols[1].strip()
  not_before = cols[2].strip() if len(cols) >= 3 else ""
  not_after = cols[3].strip() if len(cols) >= 4 else ""
  status = cols[4].strip() if len(cols) >= 5 else ""
  key_sha = cols[5].strip() if len(cols) >= 6 else ""
  cert_file_path = cols[6].strip() if len(cols) >= 7 else ""
  cert_sha = cols[7].strip() if len(cols) >= 8 else ""
  if not key_id:
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: empty key_id",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if key_id in rows:
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: duplicate key_id '{key_id}'",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if not key_file_path:
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: empty public_key_file_path",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if status and status not in {"active", "revoked"}:
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: status must be active or revoked",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if key_sha and not re.fullmatch(r"[0-9a-f]{64}", key_sha):
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: key_sha256 must be 64 hex chars",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if cert_sha and not re.fullmatch(r"[0-9a-f]{64}", cert_sha):
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: cert_sha256 must be 64 hex chars",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if cert_sha and not cert_file_path:
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: cert_sha256 requires cert_file_path",
        file=sys.stderr,
    )
    raise SystemExit(1)
  not_before = parse_date(not_before, f"row {keyring_path}:{line_no} not_before")
  not_after = parse_date(not_after, f"row {keyring_path}:{line_no} not_after")
  if not_before and not_after and not_before > not_after:
    print(
        f"invalid lane state Ed25519 keyring row {keyring_path}:{line_no}: not_before is after not_after",
        file=sys.stderr,
    )
    raise SystemExit(1)
  rows[key_id] = (
      key_file_path,
      not_before,
      not_after,
      status,
      key_sha,
      cert_file_path,
      cert_sha,
  )

if target_key_id not in rows:
  print(
      f"lane state Ed25519 keyring missing key_id '{target_key_id}' in {keyring_path}",
      file=sys.stderr,
  )
  raise SystemExit(1)

key_file_path, not_before, not_after, status, key_sha, cert_file_path, cert_sha = rows[target_key_id]
if status == "revoked":
  print(
      f"lane state Ed25519 key_id '{target_key_id}' is revoked in keyring {keyring_path}",
      file=sys.stderr,
  )
  raise SystemExit(1)

key_path = Path(key_file_path)
if not key_path.is_absolute():
  key_path = (keyring_path.parent / key_path).resolve()
if not key_path.is_file():
  print(
      f"lane state Ed25519 public key file for key_id '{target_key_id}' not found: {key_path}",
      file=sys.stderr,
  )
  raise SystemExit(1)
actual_key_sha = hashlib.sha256(key_path.read_bytes()).hexdigest()
if key_sha and actual_key_sha != key_sha:
  print(
      f"lane state Ed25519 public key SHA256 mismatch for key_id '{target_key_id}': expected {key_sha}, found {actual_key_sha}",
      file=sys.stderr,
  )
  raise SystemExit(1)

cert_sha_resolved = ""
if cert_file_path:
  cert_path = Path(cert_file_path)
  if not cert_path.is_absolute():
    cert_path = (keyring_path.parent / cert_path).resolve()
  if not cert_path.is_file():
    print(
        f"lane state Ed25519 certificate file for key_id '{target_key_id}' not found: {cert_path}",
        file=sys.stderr,
    )
    raise SystemExit(1)
  cert_bytes = cert_path.read_bytes()
  cert_sha_resolved = hashlib.sha256(cert_bytes).hexdigest()
  if cert_sha and cert_sha_resolved != cert_sha:
    print(
        f"lane state Ed25519 certificate SHA256 mismatch for key_id '{target_key_id}': expected {cert_sha}, found {cert_sha_resolved}",
        file=sys.stderr,
    )
    raise SystemExit(1)
  with tempfile.NamedTemporaryFile(delete=False) as cert_pub_file:
    cert_pub_path = cert_pub_file.name
  try:
    cert_pub = run_openssl(
        ["openssl", "x509", "-in", str(cert_path), "-pubkey", "-noout"],
        f"lane state Ed25519 certificate public key extraction failed for key_id '{target_key_id}'",
    ).stdout
    Path(cert_pub_path).write_bytes(cert_pub)
    cert_der = run_openssl(
        ["openssl", "pkey", "-pubin", "-in", cert_pub_path, "-outform", "DER"],
        f"lane state Ed25519 certificate public key conversion failed for key_id '{target_key_id}'",
    ).stdout
    key_der = run_openssl(
        ["openssl", "pkey", "-pubin", "-in", str(key_path), "-outform", "DER"],
        f"lane state Ed25519 public key conversion failed for key_id '{target_key_id}'",
    ).stdout
  finally:
    Path(cert_pub_path).unlink(missing_ok=True)
  cert_pub_sha = hashlib.sha256(cert_der).hexdigest()
  key_pub_sha = hashlib.sha256(key_der).hexdigest()
  if cert_pub_sha != key_pub_sha:
    print(
        f"lane state Ed25519 certificate/public key mismatch for key_id '{target_key_id}'",
        file=sys.stderr,
    )
    raise SystemExit(1)
  cert_subject = run_openssl(
      ["openssl", "x509", "-in", str(cert_path), "-noout", "-subject", "-nameopt", "RFC2253"],
      f"lane state Ed25519 certificate subject extraction failed for key_id '{target_key_id}'",
  ).stdout.decode("utf-8", errors="replace").strip()
  if cert_subject_regex:
    try:
      subject_match = re.search(cert_subject_regex, cert_subject) is not None
    except re.error as ex:
      print(
          f"invalid --lane-state-manifest-ed25519-cert-subject-regex: {ex}",
          file=sys.stderr,
      )
      raise SystemExit(1)
    if not subject_match:
      print(
          f"lane state Ed25519 certificate subject mismatch for key_id '{target_key_id}': regex '{cert_subject_regex}' did not match '{cert_subject}'",
          file=sys.stderr,
      )
      raise SystemExit(1)
  if ca_file:
    if crl_file:
      crl_next_update_raw = run_openssl(
          ["openssl", "crl", "-in", crl_file, "-noout", "-nextupdate", "-dateopt", "iso_8601"],
          f"lane state Ed25519 CRL nextUpdate extraction failed for key_id '{target_key_id}'",
      ).stdout.decode("utf-8", errors="replace").strip()
      if not crl_next_update_raw.startswith("nextUpdate="):
        print(
            f"lane state Ed25519 CRL nextUpdate parse failed for key_id '{target_key_id}': expected nextUpdate=... in '{crl_next_update_raw}'",
            file=sys.stderr,
        )
        raise SystemExit(1)
      crl_next_update = crl_next_update_raw.split("=", 1)[1].strip()
      try:
        crl_next_update_dt = datetime.strptime(crl_next_update, "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
      except ValueError:
        print(
            f"lane state Ed25519 CRL nextUpdate parse failed for key_id '{target_key_id}': unsupported timestamp '{crl_next_update}'",
            file=sys.stderr,
        )
        raise SystemExit(1)
      now_utc = datetime.now(timezone.utc)
      if crl_next_update_dt < now_utc:
        print(
            f"lane state Ed25519 CRL is stale for key_id '{target_key_id}': nextUpdate {crl_next_update} is before current UTC time {now_utc.strftime('%Y-%m-%d %H:%M:%SZ')}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    verify_cmd = ["openssl", "verify", "-CAfile", ca_file]
    if crl_file:
      verify_cmd.extend(["-crl_check", "-CRLfile", crl_file])
    verify_cmd.append(str(cert_path))
    run_openssl(
        verify_cmd,
        f"lane state Ed25519 certificate verify failed for key_id '{target_key_id}'",
    )
    if ocsp_response_file:
      ocsp_issuer_cert_file_effective = ocsp_issuer_cert_file if ocsp_issuer_cert_file else ca_file
      ocsp_sha = hashlib.sha256(Path(ocsp_response_file).read_bytes()).hexdigest()
      if ocsp_response_expected_sha and ocsp_sha != ocsp_response_expected_sha:
        print(
            f"lane state Ed25519 OCSP response SHA256 mismatch for key_id '{target_key_id}': expected {ocsp_response_expected_sha}, found {ocsp_sha}",
            file=sys.stderr,
        )
        raise SystemExit(1)
      ocsp_verify_args = ["-VAfile", ca_file]
      if ocsp_responder_cert_file:
        try:
          ocsp_responder_cert_sha = hashlib.sha256(Path(ocsp_responder_cert_file).read_bytes()).hexdigest()
        except OSError as ex:
          print(
              f"lane state Ed25519 OCSP responder cert read failed for key_id '{target_key_id}': {ex}",
              file=sys.stderr,
          )
          raise SystemExit(1)
        if ocsp_responder_cert_expected_sha and ocsp_responder_cert_sha != ocsp_responder_cert_expected_sha:
          print(
              f"lane state Ed25519 OCSP responder cert SHA256 mismatch for key_id '{target_key_id}': expected {ocsp_responder_cert_expected_sha}, found {ocsp_responder_cert_sha}",
              file=sys.stderr,
          )
          raise SystemExit(1)
        run_openssl(
            ["openssl", "verify", "-CAfile", ca_file, ocsp_responder_cert_file],
            f"lane state Ed25519 OCSP responder cert verify failed for key_id '{target_key_id}'",
        )
        responder_cert_text = ""
        if ocsp_require_responder_ocsp_signing or ocsp_require_responder_aki_match_ca_ski:
          responder_cert_text = run_openssl(
              ["openssl", "x509", "-in", ocsp_responder_cert_file, "-noout", "-text"],
              f"lane state Ed25519 OCSP responder cert metadata extraction failed for key_id '{target_key_id}'",
          ).stdout.decode("utf-8", errors="replace")
        if ocsp_require_responder_ocsp_signing:
          eku_match = re.search(r"X509v3 Extended Key Usage:\s*\n((?:\s+.+\n)+)", responder_cert_text)
          eku_block = eku_match.group(1) if eku_match is not None else ""
          if "OCSP Signing" not in eku_block:
            print(
                f"lane state Ed25519 OCSP responder cert EKU missing OCSP Signing for key_id '{target_key_id}'",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if ocsp_require_responder_aki_match_ca_ski:
          ca_cert_text = run_openssl(
              ["openssl", "x509", "-in", ca_file, "-noout", "-text"],
              f"lane state Ed25519 CA cert SKI extraction failed for key_id '{target_key_id}'",
          ).stdout.decode("utf-8", errors="replace")
          responder_aki_match = re.search(
              r"X509v3 Authority Key Identifier:\s*\n\s*(?:keyid:)?\s*([0-9A-Fa-f:]+)",
              responder_cert_text,
          )
          if responder_aki_match is None:
            print(
                f"lane state Ed25519 OCSP responder cert AKI keyid missing for key_id '{target_key_id}'",
                file=sys.stderr,
            )
            raise SystemExit(1)
          ca_ski_match = re.search(r"X509v3 Subject Key Identifier:\s*\n\s*([0-9A-Fa-f:]+)", ca_cert_text)
          if ca_ski_match is None:
            print(
                f"lane state Ed25519 CA cert SKI missing for key_id '{target_key_id}'",
                file=sys.stderr,
            )
            raise SystemExit(1)
          responder_aki = re.sub(r"[^0-9A-Fa-f]", "", responder_aki_match.group(1)).lower()
          ca_ski = re.sub(r"[^0-9A-Fa-f]", "", ca_ski_match.group(1)).lower()
          if responder_aki != ca_ski:
            print(
                f"lane state Ed25519 OCSP responder cert AKI/SKI mismatch for key_id '{target_key_id}': responder AKI {responder_aki} != CA SKI {ca_ski}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        ocsp_verify_args = ["-verify_other", ocsp_responder_cert_file, "-VAfile", ocsp_responder_cert_file]
      ocsp_status_raw = run_openssl(
          [
              "openssl",
              "ocsp",
              "-issuer",
              ocsp_issuer_cert_file_effective,
              "-cert",
              str(cert_path),
              "-respin",
              ocsp_response_file,
              "-CAfile",
              ca_file,
              *ocsp_verify_args,
              "-no_nonce",
          ],
          f"lane state Ed25519 OCSP verification failed for key_id '{target_key_id}'",
      ).stdout.decode("utf-8", errors="replace")
      status_match = re.search(r":\s*(good|revoked|unknown)\b", ocsp_status_raw)
      if status_match is None:
        print(
            f"lane state Ed25519 OCSP status parse failed for key_id '{target_key_id}'",
            file=sys.stderr,
        )
        raise SystemExit(1)
      ocsp_status = status_match.group(1)
      if ocsp_status != "good":
        print(
            f"lane state Ed25519 OCSP status is {ocsp_status} for key_id '{target_key_id}'",
            file=sys.stderr,
        )
        raise SystemExit(1)
      this_update_match = re.search(r"This Update:\s*(.+)", ocsp_status_raw)
      if this_update_match is None:
        print(
            f"lane state Ed25519 OCSP thisUpdate missing for key_id '{target_key_id}'",
            file=sys.stderr,
        )
        raise SystemExit(1)
      this_update_dt = parse_ocsp_time(this_update_match.group(1), "thisUpdate")
      now_utc = datetime.now(timezone.utc)
      if this_update_dt > now_utc + timedelta(minutes=5):
        print(
            f"lane state Ed25519 OCSP thisUpdate is in the future for key_id '{target_key_id}': {this_update_dt.strftime('%Y-%m-%d %H:%M:%SZ')}",
            file=sys.stderr,
        )
        raise SystemExit(1)
      if ocsp_max_age_secs:
        max_age = int(ocsp_max_age_secs)
        age_seconds = (now_utc - this_update_dt).total_seconds()
        if age_seconds > max_age:
          print(
              f"lane state Ed25519 OCSP response is stale for key_id '{target_key_id}': age {int(age_seconds)}s exceeds max {max_age}s",
              file=sys.stderr,
          )
          raise SystemExit(1)
      next_update_match = re.search(r"Next Update:\s*(.+)", ocsp_status_raw)
      if next_update_match is None and ocsp_require_next_update:
        print(
            f"lane state Ed25519 OCSP nextUpdate missing for key_id '{target_key_id}'",
            file=sys.stderr,
        )
        raise SystemExit(1)
      if next_update_match is not None:
        next_update_dt = parse_ocsp_time(next_update_match.group(1), "nextUpdate")
        if next_update_dt < now_utc:
          print(
              f"lane state Ed25519 OCSP nextUpdate is stale for key_id '{target_key_id}': {next_update_dt.strftime('%Y-%m-%d %H:%M:%SZ')}",
              file=sys.stderr,
          )
          raise SystemExit(1)
      if ocsp_responder_id_regex:
        ocsp_text = run_openssl(
            [
                "openssl",
                "ocsp",
                "-issuer",
                ocsp_issuer_cert_file_effective,
                "-cert",
                str(cert_path),
                "-respin",
                ocsp_response_file,
                "-CAfile",
                ca_file,
                *ocsp_verify_args,
                "-no_nonce",
                "-resp_text",
            ],
            f"lane state Ed25519 OCSP responder-id extraction failed for key_id '{target_key_id}'",
        ).stdout.decode("utf-8", errors="replace")
        responder_match = re.search(r"Responder Id:\s*(.+)", ocsp_text)
        if responder_match is None:
          print(
              f"lane state Ed25519 OCSP responder-id parse failed for key_id '{target_key_id}'",
              file=sys.stderr,
          )
          raise SystemExit(1)
        responder_id = responder_match.group(1).strip()
        try:
          responder_id_ok = re.search(ocsp_responder_id_regex, responder_id) is not None
        except re.error as ex:
          print(
              f"invalid --lane-state-manifest-ed25519-ocsp-responder-id-regex: {ex}",
              file=sys.stderr,
          )
          raise SystemExit(1)
        if not responder_id_ok:
          print(
              f"lane state Ed25519 OCSP responder-id mismatch for key_id '{target_key_id}': regex '{ocsp_responder_id_regex}' did not match '{responder_id}'",
              file=sys.stderr,
          )
          raise SystemExit(1)
elif ca_file:
  print(
      f"lane state Ed25519 keyring key_id '{target_key_id}' missing cert_file_path while --lane-state-manifest-ed25519-ca-file is set",
      file=sys.stderr,
  )
  raise SystemExit(1)

print(str(key_path))
print(actual_key_sha)
print(actual_sha)
print(not_before)
print(not_after)
print(cert_sha_resolved)
PY
    )
    if [[ "${#lane_state_ed25519_keyring_resolved[@]}" -ne 6 ]]; then
      echo "internal error: failed to resolve lane state Ed25519 keyring" >&2
      exit 1
    fi
    LANE_STATE_MANIFEST_ED25519_EFFECTIVE_PUBLIC_KEY_FILE="${lane_state_ed25519_keyring_resolved[0]}"
    LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_SHA256="${lane_state_ed25519_keyring_resolved[1]}"
    LANE_STATE_MANIFEST_ED25519_KEYRING_SHA256_RESOLVED="${lane_state_ed25519_keyring_resolved[2]}"
    LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_NOT_BEFORE="${lane_state_ed25519_keyring_resolved[3]}"
    LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_NOT_AFTER="${lane_state_ed25519_keyring_resolved[4]}"
    LANE_STATE_MANIFEST_ED25519_CERT_SHA256="${lane_state_ed25519_keyring_resolved[5]}"
    if [[ -n "$LANE_STATE_MANIFEST_ED25519_CA_FILE" ]]; then
      LANE_STATE_MANIFEST_ED25519_CA_SHA256="$(
        sha256sum "$LANE_STATE_MANIFEST_ED25519_CA_FILE" | awk '{print $1}'
      )"
      if [[ -n "$LANE_STATE_MANIFEST_ED25519_CRL_FILE" ]]; then
        LANE_STATE_MANIFEST_ED25519_CRL_SHA256="$(
          sha256sum "$LANE_STATE_MANIFEST_ED25519_CRL_FILE" | awk '{print $1}'
        )"
      else
        LANE_STATE_MANIFEST_ED25519_CRL_SHA256=""
      fi
      if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" ]]; then
        LANE_STATE_MANIFEST_ED25519_OCSP_SHA256="$(
          sha256sum "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONSE_FILE" | awk '{print $1}'
        )"
        if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE" ]]; then
          LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256="$(
            sha256sum "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_FILE" | awk '{print $1}'
          )"
        else
          LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256=""
        fi
        if [[ -n "$LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_FILE" ]]; then
          LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256="$(
            sha256sum "$LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_FILE" | awk '{print $1}'
          )"
        else
          LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256=""
        fi
      else
        LANE_STATE_MANIFEST_ED25519_OCSP_SHA256=""
        LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256=""
        LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256=""
      fi
    else
      LANE_STATE_MANIFEST_ED25519_CRL_SHA256=""
      LANE_STATE_MANIFEST_ED25519_OCSP_SHA256=""
      LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256=""
      LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256=""
    fi
    LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_MODE="keyring"
  else
    LANE_STATE_MANIFEST_ED25519_EFFECTIVE_PUBLIC_KEY_FILE="$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE"
    LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_SHA256="$(
      sha256sum "$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE" | awk '{print $1}'
    )"
    LANE_STATE_MANIFEST_ED25519_KEYRING_SHA256_RESOLVED=""
    LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_NOT_BEFORE=""
    LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_NOT_AFTER=""
    LANE_STATE_MANIFEST_ED25519_CERT_SHA256=""
    LANE_STATE_MANIFEST_ED25519_CA_SHA256=""
    LANE_STATE_MANIFEST_ED25519_CRL_SHA256=""
    LANE_STATE_MANIFEST_ED25519_OCSP_SHA256=""
    LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256=""
    LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256=""
    LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_MODE="file"
  fi
  LANE_STATE_MANIFEST_SIGN_MODE="ed25519"
elif [[ -n "$LANE_STATE_HMAC_KEYRING_TSV" ]]; then
  mapfile -t lane_state_keyring_resolved < <(
    LANE_STATE_HMAC_KEYRING_TSV="$LANE_STATE_HMAC_KEYRING_TSV" \
    LANE_STATE_HMAC_KEYRING_SHA256="$LANE_STATE_HMAC_KEYRING_SHA256" \
    LANE_STATE_HMAC_KEY_ID="$LANE_STATE_HMAC_KEY_ID" \
    python3 - <<'PY'
import hashlib
from datetime import datetime
import os
import re
import sys
from pathlib import Path

keyring_path = Path(os.environ["LANE_STATE_HMAC_KEYRING_TSV"])
expected_sha = os.environ.get("LANE_STATE_HMAC_KEYRING_SHA256", "").strip()
target_key_id = os.environ["LANE_STATE_HMAC_KEY_ID"].strip()

keyring_bytes = keyring_path.read_bytes()
actual_sha = hashlib.sha256(keyring_bytes).hexdigest()
if expected_sha and actual_sha != expected_sha:
  print(
      f"lane state HMAC keyring SHA256 mismatch: expected {expected_sha}, found {actual_sha}",
      file=sys.stderr,
  )
  raise SystemExit(1)

rows = {}

def parse_date(value: str, field: str) -> str:
  if not value:
    return ""
  if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
    print(
        f"invalid lane state HMAC keyring {field}: expected YYYY-MM-DD",
        file=sys.stderr,
    )
    raise SystemExit(1)
  try:
    datetime.strptime(value, "%Y-%m-%d")
  except ValueError:
    print(
        f"invalid lane state HMAC keyring {field}: invalid calendar date",
        file=sys.stderr,
    )
    raise SystemExit(1)
  return value

for line_no, raw_line in enumerate(keyring_bytes.decode("utf-8").splitlines(), start=1):
  line = raw_line.strip()
  if not line or line.startswith("#"):
    continue
  cols = raw_line.split("\t")
  if len(cols) < 2:
    print(
        f"invalid lane state HMAC keyring row {keyring_path}:{line_no}: expected at least key_id and key_file_path",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if len(cols) > 6:
    print(
        f"invalid lane state HMAC keyring row {keyring_path}:{line_no}: expected at most 6 columns",
        file=sys.stderr,
    )
    raise SystemExit(1)
  key_id = cols[0].strip()
  key_file_path = cols[1].strip()
  not_before = cols[2].strip() if len(cols) >= 3 else ""
  not_after = cols[3].strip() if len(cols) >= 4 else ""
  status = cols[4].strip() if len(cols) >= 5 else ""
  key_sha = cols[5].strip() if len(cols) >= 6 else ""
  if not key_id:
    print(
        f"invalid lane state HMAC keyring row {keyring_path}:{line_no}: empty key_id",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if key_id in rows:
    print(
        f"invalid lane state HMAC keyring row {keyring_path}:{line_no}: duplicate key_id '{key_id}'",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if not key_file_path:
    print(
        f"invalid lane state HMAC keyring row {keyring_path}:{line_no}: empty key_file_path",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if status and status not in {"active", "revoked"}:
    print(
        f"invalid lane state HMAC keyring row {keyring_path}:{line_no}: status must be active or revoked",
        file=sys.stderr,
    )
    raise SystemExit(1)
  if key_sha and not re.fullmatch(r"[0-9a-f]{64}", key_sha):
    print(
        f"invalid lane state HMAC keyring row {keyring_path}:{line_no}: key_sha256 must be 64 hex chars",
        file=sys.stderr,
    )
    raise SystemExit(1)
  not_before = parse_date(not_before, f"row {keyring_path}:{line_no} not_before")
  not_after = parse_date(not_after, f"row {keyring_path}:{line_no} not_after")
  if not_before and not_after and not_before > not_after:
    print(
        f"invalid lane state HMAC keyring row {keyring_path}:{line_no}: not_before is after not_after",
        file=sys.stderr,
    )
    raise SystemExit(1)
  rows[key_id] = (key_file_path, not_before, not_after, status, key_sha)

if target_key_id not in rows:
  print(
      f"lane state HMAC keyring missing key_id '{target_key_id}' in {keyring_path}",
      file=sys.stderr,
  )
  raise SystemExit(1)

key_file_path, not_before, not_after, status, key_sha = rows[target_key_id]
if status == "revoked":
  print(
      f"lane state HMAC key_id '{target_key_id}' is revoked in keyring {keyring_path}",
      file=sys.stderr,
  )
  raise SystemExit(1)

key_path = Path(key_file_path)
if not key_path.is_absolute():
  key_path = (keyring_path.parent / key_path).resolve()
if not key_path.is_file():
  print(
      f"lane state HMAC key file for key_id '{target_key_id}' not found: {key_path}",
      file=sys.stderr,
  )
  raise SystemExit(1)
if key_sha:
  actual_key_sha = hashlib.sha256(key_path.read_bytes()).hexdigest()
  if actual_key_sha != key_sha:
    print(
        f"lane state HMAC key SHA256 mismatch for key_id '{target_key_id}': expected {key_sha}, found {actual_key_sha}",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(str(key_path))
print(actual_sha)
print(not_before)
print(not_after)
PY
  )
  if [[ "${#lane_state_keyring_resolved[@]}" -ne 4 ]]; then
    echo "internal error: failed to resolve lane state HMAC keyring" >&2
    exit 1
  fi
  LANE_STATE_HMAC_EFFECTIVE_KEY_FILE="${lane_state_keyring_resolved[0]}"
  LANE_STATE_HMAC_KEYRING_SHA256_RESOLVED="${lane_state_keyring_resolved[1]}"
  LANE_STATE_HMAC_KEY_NOT_BEFORE="${lane_state_keyring_resolved[2]}"
  LANE_STATE_HMAC_KEY_NOT_AFTER="${lane_state_keyring_resolved[3]}"
  LANE_STATE_HMAC_MODE="keyring"
  LANE_STATE_MANIFEST_SIGN_MODE="hmac"
elif [[ -n "$LANE_STATE_HMAC_KEY_FILE" ]]; then
  LANE_STATE_HMAC_EFFECTIVE_KEY_FILE="$LANE_STATE_HMAC_KEY_FILE"
  LANE_STATE_HMAC_MODE="key_file"
  LANE_STATE_MANIFEST_SIGN_MODE="hmac"
else
  LANE_STATE_HMAC_MODE="none"
  LANE_STATE_MANIFEST_SIGN_MODE="none"
fi

mkdir -p "$OUT_DIR"

emit_expectations_dry_run_run_end() {
  local exit_code="$?"
  if [[ "$EXPECTATIONS_DRY_RUN" == "1" && -n "$EXPECTATIONS_DRY_RUN_REPORT_JSONL" ]]; then
    OUT_DIR="$OUT_DIR" \
    DATE_STR="$DATE_STR" \
    EXPECTATIONS_DRY_RUN_RUN_ID="$RUN_ID" \
    EXPECTATIONS_DRY_RUN_REPORT_JSONL="$EXPECTATIONS_DRY_RUN_REPORT_JSONL" \
    EXPECTATIONS_DRY_RUN_EXIT_CODE="$exit_code" \
    EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE="$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE" \
    EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID="$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID" \
    python3 - <<'PY'
import hmac
import hashlib
import json
import os
from pathlib import Path

report_path = Path(os.environ["EXPECTATIONS_DRY_RUN_REPORT_JSONL"])
report_path.parent.mkdir(parents=True, exist_ok=True)
run_id = os.environ.get("EXPECTATIONS_DRY_RUN_RUN_ID", "")
rows_for_run = []
if report_path.exists():
  for line in report_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      row = json.loads(line)
    except Exception:
      continue
    if row.get("run_id", "") == run_id:
      rows_for_run.append(row)

digest = hashlib.sha256()
for row in rows_for_run:
  digest.update(
      json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
  )
  digest.update(b"\n")

payload_hash_hex = digest.hexdigest()
payload = {
    "operation": "run_end",
    "schema_version": 1,
    "date": os.environ.get("DATE_STR", ""),
    "run_id": run_id,
    "out_dir": os.environ.get("OUT_DIR", ""),
    "exit_code": int(os.environ.get("EXPECTATIONS_DRY_RUN_EXIT_CODE", "0")),
    "row_count": len(rows_for_run),
    "payload_sha256": payload_hash_hex,
}
hmac_key_file = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE", "")
if hmac_key_file:
  key_bytes = Path(hmac_key_file).read_bytes()
  key_id = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID", "").strip()
  payload["payload_hmac_sha256"] = hmac.new(
      key_bytes, payload_hash_hex.encode("utf-8"), hashlib.sha256
  ).hexdigest()
  if key_id:
    payload["hmac_key_id"] = key_id
with report_path.open("a", encoding="utf-8") as f:
  f.write(json.dumps(payload, sort_keys=True) + "\n")
PY
  fi
  return "$exit_code"
}
trap emit_expectations_dry_run_run_end EXIT

if [[ "$EXPECTATIONS_DRY_RUN" == "1" && -n "$EXPECTATIONS_DRY_RUN_REPORT_JSONL" ]]; then
  OUT_DIR="$OUT_DIR" \
  DATE_STR="$DATE_STR" \
  EXPECTATIONS_DRY_RUN_RUN_ID="$RUN_ID" \
  EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS="$EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS" \
  EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE="$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE" \
  EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID="$EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID" \
  EXPECTATIONS_DRY_RUN_REPORT_JSONL="$EXPECTATIONS_DRY_RUN_REPORT_JSONL" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

report_path = Path(os.environ["EXPECTATIONS_DRY_RUN_REPORT_JSONL"])
report_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "operation": "run_meta",
    "schema_version": 1,
    "date": os.environ.get("DATE_STR", ""),
    "run_id": os.environ.get("EXPECTATIONS_DRY_RUN_RUN_ID", ""),
    "report_sample_rows_limit": int(
        os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS", "5")
    ),
    "hmac_mode": "sha256-keyfile"
    if os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_FILE", "")
    else "none",
    "out_dir": os.environ.get("OUT_DIR", ""),
}
if payload["hmac_mode"] != "none":
  hmac_key_id = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_HMAC_KEY_ID", "").strip()
  if hmac_key_id:
    payload["hmac_key_id"] = hmac_key_id
with report_path.open("a", encoding="utf-8") as f:
  f.write(json.dumps(payload, sort_keys=True) + "\n")
PY
fi

declare -A LANE_STATE_ROW_BY_ID=()
declare -A LANE_STATE_SOURCE_BY_ID=()
declare -a LANE_STATE_ORDER=()
LANE_STATE_ACTIVE_CONFIG_HASH=""
LANE_STATE_COMPAT_POLICY_VERSION="1"

compute_lane_state_config_hash() {
  local script_sha=""
  if [[ -r "$0" ]]; then
    script_sha="$(sha256sum "$0" | awk '{print $1}' || true)"
  fi
  {
    printf "schema_version=1\n"
    printf "lane_state_compat_policy_version=%s\n" "$LANE_STATE_COMPAT_POLICY_VERSION"
    printf "script_sha256=%s\n" "$script_sha"
    printf "bmc_run_smtlib=%s\n" "$BMC_RUN_SMTLIB"
    printf "bmc_allow_multi_clock=%s\n" "$BMC_ALLOW_MULTI_CLOCK"
    printf "bmc_assume_known_inputs=%s\n" "$BMC_ASSUME_KNOWN_INPUTS"
    printf "lec_assume_known_inputs=%s\n" "$LEC_ASSUME_KNOWN_INPUTS"
    printf "lec_accept_xprop_only=%s\n" "$LEC_ACCEPT_XPROP_ONLY"
    printf "with_opentitan=%s\n" "$WITH_OPENTITAN"
    printf "with_opentitan_lec_strict=%s\n" "$WITH_OPENTITAN_LEC_STRICT"
    printf "with_opentitan_e2e=%s\n" "$WITH_OPENTITAN_E2E"
    printf "with_opentitan_e2e_strict=%s\n" "$WITH_OPENTITAN_E2E_STRICT"
    printf "with_avip=%s\n" "$WITH_AVIP"
    printf "opentitan_e2e_sim_targets=%s\n" "$OPENTITAN_E2E_SIM_TARGETS"
    printf "opentitan_e2e_verilog_targets=%s\n" "$OPENTITAN_E2E_VERILOG_TARGETS"
    printf "opentitan_e2e_sim_timeout=%s\n" "$OPENTITAN_E2E_SIM_TIMEOUT"
    printf "opentitan_e2e_impl_filter=%s\n" "$OPENTITAN_E2E_IMPL_FILTER"
    printf "opentitan_e2e_include_masked=%s\n" "$OPENTITAN_E2E_INCLUDE_MASKED"
    printf "opentitan_e2e_lec_x_mode=%s\n" "$OPENTITAN_E2E_LEC_X_MODE"
    printf "z3_bin=%s\n" "$Z3_BIN"
    printf "circt_verilog=%s\n" "$CIRCT_VERILOG_BIN"
    printf "circt_verilog_avip=%s\n" "$CIRCT_VERILOG_BIN_AVIP"
    printf "circt_verilog_opentitan=%s\n" "$CIRCT_VERILOG_BIN_OPENTITAN"
    printf "sv_tests_dir=%s\n" "$SV_TESTS_DIR"
    printf "verilator_dir=%s\n" "$VERILATOR_DIR"
    printf "yosys_dir=%s\n" "$YOSYS_DIR"
    printf "opentitan_dir=%s\n" "$OPENTITAN_DIR"
    printf "avip_glob=%s\n" "$AVIP_GLOB"
    printf "lane_state_hmac_mode=%s\n" "$LANE_STATE_HMAC_MODE"
    printf "lane_state_hmac_key_id=%s\n" "$LANE_STATE_HMAC_KEY_ID"
    printf "lane_state_hmac_keyring_sha256=%s\n" "$LANE_STATE_HMAC_KEYRING_SHA256_RESOLVED"
    printf "lane_state_manifest_sign_mode=%s\n" "$LANE_STATE_MANIFEST_SIGN_MODE"
    printf "lane_state_ed25519_public_key_mode=%s\n" "$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_MODE"
    printf "lane_state_ed25519_key_id=%s\n" "$LANE_STATE_MANIFEST_ED25519_KEY_ID"
    printf "lane_state_ed25519_public_key_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_SHA256"
    printf "lane_state_ed25519_keyring_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_KEYRING_SHA256_RESOLVED"
    printf "lane_state_ed25519_cert_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_CERT_SHA256"
    printf "lane_state_ed25519_ca_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_CA_SHA256"
    printf "lane_state_ed25519_crl_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_SHA256"
    printf "lane_state_ed25519_crl_refresh_cmd=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_CMD"
    printf "lane_state_ed25519_crl_refresh_uri=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_URI"
    printf "lane_state_ed25519_crl_refresh_auto_uri_from_cert_cdp=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_FROM_CERT_CDP"
    printf "lane_state_ed25519_crl_refresh_auto_uri_policy=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_POLICY"
    printf "lane_state_ed25519_crl_refresh_auto_uri_allowed_schemes=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_AUTO_URI_ALLOWED_SCHEMES"
    printf "lane_state_ed25519_refresh_policy_profile=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILE"
    printf "lane_state_ed25519_refresh_policy_profiles_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_SHA256_RESOLVED"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_SHA256_RESOLVED"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_public_key_mode=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_MODE"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_public_key_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_SHA256"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_SHA256_RESOLVED"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ca_file=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_FILE"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ca_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CA_SHA256"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_crl_file=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_FILE"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_crl_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_CRL_SHA256"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_response_file=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_FILE"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_response_sha256_expected=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONSE_SHA256"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_response_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_SHA256"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_max_age_secs=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_MAX_AGE_SECS"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_require_next_update=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_NEXT_UPDATE"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_responder_cert_file=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_FILE"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_issuer_cert_file=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_FILE"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_responder_cert_sha256_expected=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_SHA256_EXPECTED"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_responder_cert_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_CERT_SHA256"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_issuer_cert_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_ISSUER_CERT_SHA256"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_require_responder_ocsp_signing=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_RESPONDER_OCSP_SIGNING"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_require_responder_aki_match_ca_ski=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_REQUIRE_RESPONDER_AKI_MATCH_CA_SKI"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_keyring_ocsp_responder_id_regex=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEYRING_OCSP_RESPONDER_ID_REGEX"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_public_key_not_before=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_NOT_BEFORE"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_public_key_not_after=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_PUBLIC_KEY_NOT_AFTER"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_cert_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_CERT_SHA256"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_key_id=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEY_ID_RESOLVED"
    printf "lane_state_ed25519_refresh_policy_profiles_manifest_expected_key_id=%s\n" "$LANE_STATE_MANIFEST_ED25519_REFRESH_POLICY_PROFILES_MANIFEST_KEY_ID"
    printf "lane_state_ed25519_crl_refresh_retries=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_RETRIES"
    printf "lane_state_ed25519_crl_refresh_delay_secs=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_DELAY_SECS"
    printf "lane_state_ed25519_crl_refresh_timeout_secs=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_TIMEOUT_SECS"
    printf "lane_state_ed25519_crl_refresh_jitter_secs=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_JITTER_SECS"
    printf "lane_state_ed25519_crl_refresh_metadata_file=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_FILE"
    printf "lane_state_ed25519_crl_refresh_metadata_require_transport=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TRANSPORT"
    printf "lane_state_ed25519_crl_refresh_metadata_require_status=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_STATUS"
    printf "lane_state_ed25519_crl_refresh_metadata_require_uri_regex=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_URI_REGEX"
    printf "lane_state_ed25519_crl_refresh_metadata_require_tls_peer_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256"
    printf "lane_state_ed25519_crl_refresh_metadata_require_cert_chain_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256"
    printf "lane_state_ed25519_crl_refresh_metadata_require_artifact_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256"
    printf "lane_state_ed25519_crl_refresh_metadata_require_ca_cert_in_cert_chain=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN"
    printf "lane_state_ed25519_crl_refresh_metadata_require_tls_peer_in_cert_chain=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN"
    printf "lane_state_ed25519_crl_refresh_metadata_require_cert_chain_length_min=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN"
    printf "lane_state_ed25519_crl_refresh_metadata_max_age_secs=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_AGE_SECS"
    printf "lane_state_ed25519_crl_refresh_metadata_max_future_skew_secs=%s\n" "$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS"
    printf "lane_state_ed25519_ocsp_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_SHA256"
    printf "lane_state_ed25519_ocsp_refresh_cmd=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_CMD"
    printf "lane_state_ed25519_ocsp_refresh_uri=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_URI"
    printf "lane_state_ed25519_ocsp_refresh_auto_uri_from_cert_aia=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_FROM_CERT_AIA"
    printf "lane_state_ed25519_ocsp_refresh_auto_uri_policy=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_POLICY"
    printf "lane_state_ed25519_ocsp_refresh_auto_uri_allowed_schemes=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_AUTO_URI_ALLOWED_SCHEMES"
    printf "lane_state_ed25519_ocsp_refresh_retries=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_RETRIES"
    printf "lane_state_ed25519_ocsp_refresh_delay_secs=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_DELAY_SECS"
    printf "lane_state_ed25519_ocsp_refresh_timeout_secs=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_TIMEOUT_SECS"
    printf "lane_state_ed25519_ocsp_refresh_jitter_secs=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_JITTER_SECS"
    printf "lane_state_ed25519_ocsp_refresh_metadata_file=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_FILE"
    printf "lane_state_ed25519_ocsp_refresh_metadata_require_transport=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TRANSPORT"
    printf "lane_state_ed25519_ocsp_refresh_metadata_require_status=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_STATUS"
    printf "lane_state_ed25519_ocsp_refresh_metadata_require_uri_regex=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_URI_REGEX"
    printf "lane_state_ed25519_ocsp_refresh_metadata_require_tls_peer_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_SHA256"
    printf "lane_state_ed25519_ocsp_refresh_metadata_require_cert_chain_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_SHA256"
    printf "lane_state_ed25519_ocsp_refresh_metadata_require_artifact_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_ARTIFACT_SHA256"
    printf "lane_state_ed25519_ocsp_refresh_metadata_require_ca_cert_in_cert_chain=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CA_CERT_IN_CERT_CHAIN"
    printf "lane_state_ed25519_ocsp_refresh_metadata_require_tls_peer_in_cert_chain=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_TLS_PEER_IN_CERT_CHAIN"
    printf "lane_state_ed25519_ocsp_refresh_metadata_require_cert_chain_length_min=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_REQUIRE_CERT_CHAIN_LENGTH_MIN"
    printf "lane_state_ed25519_ocsp_refresh_metadata_max_age_secs=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_AGE_SECS"
    printf "lane_state_ed25519_ocsp_refresh_metadata_max_future_skew_secs=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_METADATA_MAX_FUTURE_SKEW_SECS"
    printf "lane_state_ed25519_ocsp_responder_cert_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256"
    printf "lane_state_ed25519_ocsp_issuer_cert_sha256=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256"
    printf "lane_state_ed25519_ocsp_require_responder_ocsp_signing=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_OCSP_SIGNING"
    printf "lane_state_ed25519_ocsp_require_responder_aki_match_ca_ski=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_RESPONDER_AKI_MATCH_CA_SKI"
    printf "lane_state_ed25519_ocsp_max_age_secs=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_MAX_AGE_SECS_EFFECTIVE"
    printf "lane_state_ed25519_ocsp_require_next_update=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_REQUIRE_NEXT_UPDATE"
    printf "lane_state_ed25519_ocsp_responder_id_regex=%s\n" "$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_ID_REGEX"
    printf "lane_state_ed25519_cert_subject_regex=%s\n" "$LANE_STATE_MANIFEST_ED25519_CERT_SUBJECT_REGEX"
    printf "test_filter=%s\n" "${TEST_FILTER:-}"
    printf "bmc_smoke_only=%s\n" "${BMC_SMOKE_ONLY:-}"
    printf "bmc_allow_multi_clock=%s\n" "$BMC_ALLOW_MULTI_CLOCK"
    printf "lec_smoke_only=%s\n" "${LEC_SMOKE_ONLY:-}"
    printf "circt_bmc_args=%s\n" "${CIRCT_BMC_ARGS:-}"
    printf "circt_lec_args=%s\n" "${CIRCT_LEC_ARGS:-}"
  } | sha256sum | awk '{print $1}'
}

lane_state_register_row() {
  local lane_id="$1"
  local row="$2"
  local source_ref="$3"
  if [[ -z "${LANE_STATE_ROW_BY_ID[$lane_id]+_}" ]]; then
    LANE_STATE_ORDER+=("$lane_id")
  fi
  LANE_STATE_ROW_BY_ID["$lane_id"]="$row"
  LANE_STATE_SOURCE_BY_ID["$lane_id"]="$source_ref"
}

lane_state_merge_row() {
  local lane_id="$1"
  local row="$2"
  local source_ref="$3"
  local existing_row="${LANE_STATE_ROW_BY_ID[$lane_id]-}"
  if [[ -z "$existing_row" ]]; then
    lane_state_register_row "$lane_id" "$row" "$source_ref"
    return
  fi
  local old_lane old_suite old_mode old_total old_pass old_fail old_xfail old_xpass old_error old_skip old_updated_at_utc old_compat_policy_version old_config_hash
  IFS=$'\t' read -r old_lane old_suite old_mode old_total old_pass old_fail old_xfail old_xpass old_error old_skip old_updated_at_utc old_compat_policy_version old_config_hash <<< "$existing_row"
  local new_lane new_suite new_mode new_total new_pass new_fail new_xfail new_xpass new_error new_skip new_updated_at_utc new_compat_policy_version new_config_hash
  IFS=$'\t' read -r new_lane new_suite new_mode new_total new_pass new_fail new_xfail new_xpass new_error new_skip new_updated_at_utc new_compat_policy_version new_config_hash <<< "$row"

  if [[ "$old_compat_policy_version" == "legacy" && -n "$new_compat_policy_version" && "$new_compat_policy_version" != "legacy" ]]; then
    lane_state_register_row "$lane_id" "$row" "$source_ref"
    return
  fi
  if [[ "$new_compat_policy_version" == "legacy" && -n "$old_compat_policy_version" && "$old_compat_policy_version" != "legacy" ]]; then
    return
  fi

  if [[ -n "$old_compat_policy_version" && -n "$new_compat_policy_version" && "$old_compat_policy_version" != "$new_compat_policy_version" ]]; then
    echo "conflicting lane state compatibility policy version for lane ${lane_id}: ${LANE_STATE_SOURCE_BY_ID[$lane_id]} has ${old_compat_policy_version}, ${source_ref} has ${new_compat_policy_version}" >&2
    exit 1
  fi

  if [[ -z "$old_compat_policy_version" && -n "$new_compat_policy_version" ]]; then
    lane_state_register_row "$lane_id" "$row" "$source_ref"
    return
  fi
  if [[ -n "$old_compat_policy_version" && -z "$new_compat_policy_version" ]]; then
    return
  fi

  if [[ -n "$old_config_hash" && -n "$new_config_hash" && "$old_config_hash" != "$new_config_hash" ]]; then
    echo "conflicting lane state config hash for lane ${lane_id}: ${LANE_STATE_SOURCE_BY_ID[$lane_id]} has ${old_config_hash}, ${source_ref} has ${new_config_hash}" >&2
    exit 1
  fi

  if [[ -z "$old_config_hash" && -n "$new_config_hash" ]]; then
    lane_state_register_row "$lane_id" "$row" "$source_ref"
    return
  fi
  if [[ -n "$old_config_hash" && -z "$new_config_hash" ]]; then
    return
  fi

  if [[ "$new_updated_at_utc" > "$old_updated_at_utc" ]]; then
    lane_state_register_row "$lane_id" "$row" "$source_ref"
    return
  fi
  if [[ "$new_updated_at_utc" < "$old_updated_at_utc" ]]; then
    return
  fi

  if [[ "$existing_row" != "$row" ]]; then
    echo "conflicting lane state row for lane ${lane_id}: ${LANE_STATE_SOURCE_BY_ID[$lane_id]} and ${source_ref} share updated_at_utc=${new_updated_at_utc} with different payloads" >&2
    exit 1
  fi
}

lane_state_write_file() {
  if [[ -z "$LANE_STATE_TSV" ]]; then
    return
  fi
  local lane_state_dir
  lane_state_dir="$(dirname "$LANE_STATE_TSV")"
  mkdir -p "$lane_state_dir"
  local tmp="${LANE_STATE_TSV}.tmp.$$"
  {
    printf "lane_id\tsuite\tmode\ttotal\tpass\tfail\txfail\txpass\terror\tskip\tupdated_at_utc\tcompat_policy_version\tconfig_hash\n"
    for lane_id in "${LANE_STATE_ORDER[@]}"; do
      printf "%s\n" "${LANE_STATE_ROW_BY_ID[$lane_id]}"
    done
  } > "$tmp"
  mv "$tmp" "$LANE_STATE_TSV"
  lane_state_emit_manifest "$LANE_STATE_TSV"
}

lane_state_emit_manifest() {
  local lane_state_file="$1"
  if [[ "$LANE_STATE_MANIFEST_SIGN_MODE" == "none" ]]; then
    return
  fi
  local manifest_file="${lane_state_file}.manifest.json"
  local lane_state_sha
  lane_state_sha="$(sha256sum "$lane_state_file" | awk '{print $1}')"
  LANE_STATE_MANIFEST_PATH="$manifest_file" \
  LANE_STATE_FILE="$lane_state_file" \
  LANE_STATE_SHA256="$lane_state_sha" \
  LANE_STATE_HMAC_KEY_FILE="$LANE_STATE_HMAC_EFFECTIVE_KEY_FILE" \
  LANE_STATE_HMAC_KEY_ID="$LANE_STATE_HMAC_KEY_ID" \
  LANE_STATE_HMAC_KEY_NOT_BEFORE="$LANE_STATE_HMAC_KEY_NOT_BEFORE" \
  LANE_STATE_HMAC_KEY_NOT_AFTER="$LANE_STATE_HMAC_KEY_NOT_AFTER" \
  LANE_STATE_MANIFEST_SIGN_MODE="$LANE_STATE_MANIFEST_SIGN_MODE" \
  LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE="$LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE" \
  LANE_STATE_MANIFEST_ED25519_KEY_ID="$LANE_STATE_MANIFEST_ED25519_KEY_ID" \
  LANE_STATE_MANIFEST_ED25519_CERT_SHA256="$LANE_STATE_MANIFEST_ED25519_CERT_SHA256" \
  LANE_STATE_MANIFEST_ED25519_CRL_SHA256="$LANE_STATE_MANIFEST_ED25519_CRL_SHA256" \
  LANE_STATE_MANIFEST_ED25519_OCSP_SHA256="$LANE_STATE_MANIFEST_ED25519_OCSP_SHA256" \
  LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256="$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256" \
  LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256="$LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256" \
  LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_PROVENANCE_JSON="$LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_PROVENANCE_JSON" \
  LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_PROVENANCE_JSON="$LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_PROVENANCE_JSON" \
  LANE_STATE_MANIFEST_ED25519_KEY_NOT_BEFORE="$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_NOT_BEFORE" \
  LANE_STATE_MANIFEST_ED25519_KEY_NOT_AFTER="$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_NOT_AFTER" \
  python3 - <<'PY'
import base64
import hashlib
import hmac
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

def parse_window_date(value: str, field: str, key_kind: str):
  if not value:
    return None
  try:
    return datetime.strptime(value, "%Y-%m-%d").date()
  except ValueError:
    print(f"invalid lane state {key_kind} {field}: expected YYYY-MM-DD", file=os.sys.stderr)
    raise SystemExit(1)

generated_dt = datetime.now(timezone.utc)
generated_at = generated_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
generated_date = generated_dt.date()
key_not_before = os.environ.get("LANE_STATE_HMAC_KEY_NOT_BEFORE", "").strip()
key_not_after = os.environ.get("LANE_STATE_HMAC_KEY_NOT_AFTER", "").strip()
window_start = parse_window_date(key_not_before, "not_before", "HMAC")
window_end = parse_window_date(key_not_after, "not_after", "HMAC")
if window_start and generated_date < window_start:
  print(
      f"lane state HMAC key_id '{os.environ.get('LANE_STATE_HMAC_KEY_ID', '')}' not active at generated_at_utc {generated_at} (window {key_not_before}..{key_not_after or '-'})",
      file=os.sys.stderr,
  )
  raise SystemExit(1)
if window_end and generated_date > window_end:
  print(
      f"lane state HMAC key_id '{os.environ.get('LANE_STATE_HMAC_KEY_ID', '')}' not active at generated_at_utc {generated_at} (window {key_not_before or '-'}..{key_not_after})",
      file=os.sys.stderr,
  )
  raise SystemExit(1)

payload = {
    "schema_version": 1,
    "generated_at_utc": generated_at,
    "lane_state_file": os.environ["LANE_STATE_FILE"],
    "lane_state_sha256": os.environ["LANE_STATE_SHA256"],
}
sign_mode = os.environ["LANE_STATE_MANIFEST_SIGN_MODE"]
if sign_mode == "hmac":
  payload["signature_mode"] = "hmac_sha256"
  key_id = os.environ.get("LANE_STATE_HMAC_KEY_ID", "").strip()
  if key_id:
    payload["hmac_key_id"] = key_id
  key_bytes = Path(os.environ["LANE_STATE_HMAC_KEY_FILE"]).read_bytes()
  canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
  payload["signature_hmac_sha256"] = hmac.new(
      key_bytes, canonical_payload, hashlib.sha256
  ).hexdigest()
elif sign_mode == "ed25519":
  payload["signature_mode"] = "ed25519"
  key_id = os.environ.get("LANE_STATE_MANIFEST_ED25519_KEY_ID", "").strip()
  if key_id:
    payload["ed25519_key_id"] = key_id
  cert_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_CERT_SHA256", "").strip()
  if cert_sha:
    payload["ed25519_cert_sha256"] = cert_sha
  crl_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_CRL_SHA256", "").strip()
  if crl_sha:
    payload["ed25519_crl_sha256"] = crl_sha
  ocsp_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_SHA256", "").strip()
  if ocsp_sha:
    payload["ed25519_ocsp_sha256"] = ocsp_sha
  responder_cert_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256", "").strip()
  if responder_cert_sha:
    payload["ed25519_ocsp_responder_cert_sha256"] = responder_cert_sha
  issuer_cert_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256", "").strip()
  if issuer_cert_sha:
    payload["ed25519_ocsp_issuer_cert_sha256"] = issuer_cert_sha
  crl_refresh_provenance = os.environ.get("LANE_STATE_MANIFEST_ED25519_CRL_REFRESH_PROVENANCE_JSON", "").strip()
  if crl_refresh_provenance:
    try:
      parsed_crl_refresh_provenance = json.loads(crl_refresh_provenance)
    except Exception as ex:
      print(
          f"invalid lane-state Ed25519 CRL refresh provenance payload: {ex}",
          file=os.sys.stderr,
      )
      raise SystemExit(1)
    if not isinstance(parsed_crl_refresh_provenance, dict):
      print(
          "invalid lane-state Ed25519 CRL refresh provenance payload: expected JSON object",
          file=os.sys.stderr,
      )
      raise SystemExit(1)
    payload["ed25519_crl_refresh_provenance"] = parsed_crl_refresh_provenance
  ocsp_refresh_provenance = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_REFRESH_PROVENANCE_JSON", "").strip()
  if ocsp_refresh_provenance:
    try:
      parsed_ocsp_refresh_provenance = json.loads(ocsp_refresh_provenance)
    except Exception as ex:
      print(
          f"invalid lane-state Ed25519 OCSP refresh provenance payload: {ex}",
          file=os.sys.stderr,
      )
      raise SystemExit(1)
    if not isinstance(parsed_ocsp_refresh_provenance, dict):
      print(
          "invalid lane-state Ed25519 OCSP refresh provenance payload: expected JSON object",
          file=os.sys.stderr,
      )
      raise SystemExit(1)
    payload["ed25519_ocsp_refresh_provenance"] = parsed_ocsp_refresh_provenance
  ed_key_not_before = os.environ.get("LANE_STATE_MANIFEST_ED25519_KEY_NOT_BEFORE", "").strip()
  ed_key_not_after = os.environ.get("LANE_STATE_MANIFEST_ED25519_KEY_NOT_AFTER", "").strip()
  ed_window_start = parse_window_date(ed_key_not_before, "not_before", "Ed25519")
  ed_window_end = parse_window_date(ed_key_not_after, "not_after", "Ed25519")
  if ed_window_start and generated_date < ed_window_start:
    print(
        f"lane state Ed25519 key_id '{key_id}' not active at generated_at_utc {generated_at} (window {ed_key_not_before}..{ed_key_not_after or '-'})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  if ed_window_end and generated_date > ed_window_end:
    print(
        f"lane state Ed25519 key_id '{key_id}' not active at generated_at_utc {generated_at} (window {ed_key_not_before or '-'}..{ed_key_not_after})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
  with tempfile.NamedTemporaryFile(delete=False) as payload_file:
    payload_file.write(canonical_payload)
    payload_path = payload_file.name
  with tempfile.NamedTemporaryFile(delete=False) as signature_file:
    signature_path = signature_file.name
  try:
    subprocess.run(
        [
            "openssl",
            "pkeyutl",
            "-sign",
            "-inkey",
            os.environ["LANE_STATE_MANIFEST_ED25519_PRIVATE_KEY_FILE"],
            "-rawin",
            "-in",
            payload_path,
            "-out",
            signature_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
  except subprocess.CalledProcessError as ex:
    stderr = ex.stderr.decode("utf-8", errors="replace").strip()
    print(
        f"failed to sign lane-state manifest with Ed25519 key: {stderr or ex}",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  payload["signature_ed25519_base64"] = base64.b64encode(
      Path(signature_path).read_bytes()
  ).decode("ascii")
  Path(payload_path).unlink(missing_ok=True)
  Path(signature_path).unlink(missing_ok=True)
else:
  print(f"invalid lane-state manifest signing mode: {sign_mode}", file=os.sys.stderr)
  raise SystemExit(1)

Path(os.environ["LANE_STATE_MANIFEST_PATH"]).write_text(
    json.dumps(payload, sort_keys=True), encoding="utf-8"
)
PY
}

lane_state_verify_manifest() {
  local lane_state_file="$1"
  local source_label="$2"
  if [[ "$LANE_STATE_MANIFEST_SIGN_MODE" == "none" ]]; then
    return
  fi
  local manifest_file="${lane_state_file}.manifest.json"
  if [[ ! -r "$manifest_file" ]]; then
    echo "missing lane-state manifest for ${source_label}: ${manifest_file}" >&2
    exit 1
  fi
  local lane_state_sha
  lane_state_sha="$(sha256sum "$lane_state_file" | awk '{print $1}')"
  LANE_STATE_MANIFEST_PATH="$manifest_file" \
  LANE_STATE_FILE="$lane_state_file" \
  LANE_STATE_SHA256="$lane_state_sha" \
  LANE_STATE_HMAC_KEY_FILE="$LANE_STATE_HMAC_EFFECTIVE_KEY_FILE" \
  LANE_STATE_HMAC_KEY_ID="$LANE_STATE_HMAC_KEY_ID" \
  LANE_STATE_HMAC_KEY_NOT_BEFORE="$LANE_STATE_HMAC_KEY_NOT_BEFORE" \
  LANE_STATE_HMAC_KEY_NOT_AFTER="$LANE_STATE_HMAC_KEY_NOT_AFTER" \
  LANE_STATE_MANIFEST_SIGN_MODE="$LANE_STATE_MANIFEST_SIGN_MODE" \
  LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE="$LANE_STATE_MANIFEST_ED25519_EFFECTIVE_PUBLIC_KEY_FILE" \
  LANE_STATE_MANIFEST_ED25519_KEY_ID="$LANE_STATE_MANIFEST_ED25519_KEY_ID" \
  LANE_STATE_MANIFEST_ED25519_CERT_SHA256="$LANE_STATE_MANIFEST_ED25519_CERT_SHA256" \
  LANE_STATE_MANIFEST_ED25519_CRL_SHA256="$LANE_STATE_MANIFEST_ED25519_CRL_SHA256" \
  LANE_STATE_MANIFEST_ED25519_OCSP_SHA256="$LANE_STATE_MANIFEST_ED25519_OCSP_SHA256" \
  LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256="$LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256" \
  LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256="$LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256" \
  LANE_STATE_MANIFEST_ED25519_KEY_NOT_BEFORE="$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_NOT_BEFORE" \
  LANE_STATE_MANIFEST_ED25519_KEY_NOT_AFTER="$LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_NOT_AFTER" \
  SOURCE_LABEL="$source_label" \
  python3 - <<'PY'
import base64
import hashlib
import hmac
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

source = os.environ["SOURCE_LABEL"]
manifest_path = Path(os.environ["LANE_STATE_MANIFEST_PATH"])
expected_sha = os.environ["LANE_STATE_SHA256"]
expected_file = os.environ["LANE_STATE_FILE"]
expected_key_id = os.environ.get("LANE_STATE_HMAC_KEY_ID", "").strip()
key_not_before = os.environ.get("LANE_STATE_HMAC_KEY_NOT_BEFORE", "").strip()
key_not_after = os.environ.get("LANE_STATE_HMAC_KEY_NOT_AFTER", "").strip()
expected_sign_mode = os.environ["LANE_STATE_MANIFEST_SIGN_MODE"]
expected_ed25519_key_id = os.environ.get("LANE_STATE_MANIFEST_ED25519_KEY_ID", "").strip()
expected_ed25519_cert_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_CERT_SHA256", "").strip()
expected_ed25519_crl_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_CRL_SHA256", "").strip()
expected_ed25519_ocsp_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_SHA256", "").strip()
expected_ed25519_ocsp_responder_cert_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_RESPONDER_CERT_SHA256", "").strip()
expected_ed25519_ocsp_issuer_cert_sha = os.environ.get("LANE_STATE_MANIFEST_ED25519_OCSP_ISSUER_CERT_SHA256", "").strip()
ed25519_key_not_before = os.environ.get("LANE_STATE_MANIFEST_ED25519_KEY_NOT_BEFORE", "").strip()
ed25519_key_not_after = os.environ.get("LANE_STATE_MANIFEST_ED25519_KEY_NOT_AFTER", "").strip()

def parse_window_date(value: str, field: str, key_kind: str):
  if not value:
    return None
  try:
    return datetime.strptime(value, "%Y-%m-%d").date()
  except ValueError:
    print(f"invalid lane state {key_kind} {field}: expected YYYY-MM-DD", file=os.sys.stderr)
    raise SystemExit(1)

try:
  payload = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception as ex:
  print(f"invalid lane-state manifest for {source}: unable to parse '{manifest_path}' ({ex})", file=os.sys.stderr)
  raise SystemExit(1)

if not isinstance(payload, dict):
  print(f"invalid lane-state manifest for {source}: expected JSON object", file=os.sys.stderr)
  raise SystemExit(1)

schema_version = payload.get("schema_version")
if schema_version != 1:
  print(f"invalid lane-state manifest for {source}: schema_version must be 1", file=os.sys.stderr)
  raise SystemExit(1)

manifest_file = payload.get("lane_state_file")
if not isinstance(manifest_file, str) or not manifest_file:
  print(f"invalid lane-state manifest for {source}: lane_state_file must be non-empty string", file=os.sys.stderr)
  raise SystemExit(1)
if manifest_file != expected_file:
  print(
      f"invalid lane-state manifest for {source}: lane_state_file mismatch (expected '{expected_file}', found '{manifest_file}')",
      file=os.sys.stderr,
  )
  raise SystemExit(1)

manifest_sha = payload.get("lane_state_sha256")
if not isinstance(manifest_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
  print(f"invalid lane-state manifest for {source}: lane_state_sha256 must be 64 hex chars", file=os.sys.stderr)
  raise SystemExit(1)
if manifest_sha != expected_sha:
  print(
      f"invalid lane-state manifest for {source}: lane_state_sha256 mismatch (expected {expected_sha}, found {manifest_sha})",
      file=os.sys.stderr,
  )
  raise SystemExit(1)

generated_at = payload.get("generated_at_utc")
if not isinstance(generated_at, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", generated_at):
  print(f"invalid lane-state manifest for {source}: generated_at_utc must be UTC RFC3339 timestamp", file=os.sys.stderr)
  raise SystemExit(1)
manifest_sign_mode = payload.get("signature_mode")

if expected_sign_mode == "hmac":
  if manifest_sign_mode not in (None, "", "hmac_sha256"):
    print(
        f"invalid lane-state manifest for {source}: signature_mode mismatch (expected hmac_sha256, found {manifest_sign_mode})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  generated_date = datetime.strptime(generated_at, "%Y-%m-%dT%H:%M:%SZ").date()
  window_start = parse_window_date(key_not_before, "not_before", "HMAC")
  window_end = parse_window_date(key_not_after, "not_after", "HMAC")
  if window_start and generated_date < window_start:
    print(
        f"lane state HMAC key_id '{expected_key_id}' not active at generated_at_utc {generated_at} (window {key_not_before}..{key_not_after or '-'})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  if window_end and generated_date > window_end:
    print(
        f"lane state HMAC key_id '{expected_key_id}' not active at generated_at_utc {generated_at} (window {key_not_before or '-'}..{key_not_after})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)

  signature = payload.get("signature_hmac_sha256")
  if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
    print(f"invalid lane-state manifest for {source}: signature_hmac_sha256 must be 64 hex chars", file=os.sys.stderr)
    raise SystemExit(1)

  manifest_key_id = payload.get("hmac_key_id", "")
  if manifest_key_id and not isinstance(manifest_key_id, str):
    print(f"invalid lane-state manifest for {source}: hmac_key_id must be string", file=os.sys.stderr)
    raise SystemExit(1)
  if expected_key_id and manifest_key_id != expected_key_id:
    print(
        f"invalid lane-state manifest for {source}: hmac_key_id mismatch (expected '{expected_key_id}', found '{manifest_key_id}')",
        file=os.sys.stderr,
    )
    raise SystemExit(1)

  sig_payload = dict(payload)
  del sig_payload["signature_hmac_sha256"]
  canonical_payload = json.dumps(sig_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
  key_bytes = Path(os.environ["LANE_STATE_HMAC_KEY_FILE"]).read_bytes()
  expected_sig = hmac.new(key_bytes, canonical_payload, hashlib.sha256).hexdigest()
  if signature != expected_sig:
    print(f"invalid lane-state manifest for {source}: signature_hmac_sha256 mismatch", file=os.sys.stderr)
    raise SystemExit(1)
elif expected_sign_mode == "ed25519":
  if manifest_sign_mode != "ed25519":
    print(
        f"invalid lane-state manifest for {source}: signature_mode mismatch (expected ed25519, found {manifest_sign_mode})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  manifest_key_id = payload.get("ed25519_key_id", "")
  if manifest_key_id and not isinstance(manifest_key_id, str):
    print(f"invalid lane-state manifest for {source}: ed25519_key_id must be string", file=os.sys.stderr)
    raise SystemExit(1)
  if expected_ed25519_key_id and manifest_key_id != expected_ed25519_key_id:
    print(
        f"invalid lane-state manifest for {source}: ed25519_key_id mismatch (expected '{expected_ed25519_key_id}', found '{manifest_key_id}')",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  manifest_cert_sha = payload.get("ed25519_cert_sha256", "")
  if manifest_cert_sha and (not isinstance(manifest_cert_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_cert_sha)):
    print(
        f"invalid lane-state manifest for {source}: ed25519_cert_sha256 must be 64 hex chars",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  if expected_ed25519_cert_sha and manifest_cert_sha != expected_ed25519_cert_sha:
    print(
        f"invalid lane-state manifest for {source}: ed25519_cert_sha256 mismatch (expected {expected_ed25519_cert_sha}, found {manifest_cert_sha})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  manifest_crl_sha = payload.get("ed25519_crl_sha256", "")
  if manifest_crl_sha and (not isinstance(manifest_crl_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_crl_sha)):
    print(
        f"invalid lane-state manifest for {source}: ed25519_crl_sha256 must be 64 hex chars",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  if expected_ed25519_crl_sha and manifest_crl_sha != expected_ed25519_crl_sha:
    print(
        f"invalid lane-state manifest for {source}: ed25519_crl_sha256 mismatch (expected {expected_ed25519_crl_sha}, found {manifest_crl_sha})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  manifest_ocsp_sha = payload.get("ed25519_ocsp_sha256", "")
  if manifest_ocsp_sha and (not isinstance(manifest_ocsp_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_ocsp_sha)):
    print(
        f"invalid lane-state manifest for {source}: ed25519_ocsp_sha256 must be 64 hex chars",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  if expected_ed25519_ocsp_sha and manifest_ocsp_sha != expected_ed25519_ocsp_sha:
    print(
        f"invalid lane-state manifest for {source}: ed25519_ocsp_sha256 mismatch (expected {expected_ed25519_ocsp_sha}, found {manifest_ocsp_sha})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  manifest_ocsp_responder_cert_sha = payload.get("ed25519_ocsp_responder_cert_sha256", "")
  if manifest_ocsp_responder_cert_sha and (not isinstance(manifest_ocsp_responder_cert_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_ocsp_responder_cert_sha)):
    print(
        f"invalid lane-state manifest for {source}: ed25519_ocsp_responder_cert_sha256 must be 64 hex chars",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  if expected_ed25519_ocsp_responder_cert_sha and manifest_ocsp_responder_cert_sha != expected_ed25519_ocsp_responder_cert_sha:
    print(
        f"invalid lane-state manifest for {source}: ed25519_ocsp_responder_cert_sha256 mismatch (expected {expected_ed25519_ocsp_responder_cert_sha}, found {manifest_ocsp_responder_cert_sha})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  manifest_ocsp_issuer_cert_sha = payload.get("ed25519_ocsp_issuer_cert_sha256", "")
  if manifest_ocsp_issuer_cert_sha and (not isinstance(manifest_ocsp_issuer_cert_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_ocsp_issuer_cert_sha)):
    print(
        f"invalid lane-state manifest for {source}: ed25519_ocsp_issuer_cert_sha256 must be 64 hex chars",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  if expected_ed25519_ocsp_issuer_cert_sha and manifest_ocsp_issuer_cert_sha != expected_ed25519_ocsp_issuer_cert_sha:
    print(
        f"invalid lane-state manifest for {source}: ed25519_ocsp_issuer_cert_sha256 mismatch (expected {expected_ed25519_ocsp_issuer_cert_sha}, found {manifest_ocsp_issuer_cert_sha})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  generated_date = datetime.strptime(generated_at, "%Y-%m-%dT%H:%M:%SZ").date()
  ed_window_start = parse_window_date(ed25519_key_not_before, "not_before", "Ed25519")
  ed_window_end = parse_window_date(ed25519_key_not_after, "not_after", "Ed25519")
  key_id_for_window = expected_ed25519_key_id or manifest_key_id
  if ed_window_start and generated_date < ed_window_start:
    print(
        f"lane state Ed25519 key_id '{key_id_for_window}' not active at generated_at_utc {generated_at} (window {ed25519_key_not_before}..{ed25519_key_not_after or '-'})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  if ed_window_end and generated_date > ed_window_end:
    print(
        f"lane state Ed25519 key_id '{key_id_for_window}' not active at generated_at_utc {generated_at} (window {ed25519_key_not_before or '-'}..{ed25519_key_not_after})",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  signature_b64 = payload.get("signature_ed25519_base64")
  if not isinstance(signature_b64, str) or not signature_b64.strip():
    print(
        f"invalid lane-state manifest for {source}: signature_ed25519_base64 must be non-empty string",
        file=os.sys.stderr,
    )
    raise SystemExit(1)
  try:
    signature_bytes = base64.b64decode(signature_b64.encode("ascii"), validate=True)
  except Exception:
    print(
        f"invalid lane-state manifest for {source}: signature_ed25519_base64 is not valid base64",
        file=os.sys.stderr,
    )
    raise SystemExit(1)

  sig_payload = dict(payload)
  del sig_payload["signature_ed25519_base64"]
  canonical_payload = json.dumps(sig_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
  with tempfile.NamedTemporaryFile(delete=False) as payload_file:
    payload_file.write(canonical_payload)
    payload_path = payload_file.name
  with tempfile.NamedTemporaryFile(delete=False) as signature_file:
    signature_file.write(signature_bytes)
    signature_path = signature_file.name
  try:
    verify_result = subprocess.run(
        [
            "openssl",
            "pkeyutl",
            "-verify",
            "-pubin",
            "-inkey",
            os.environ["LANE_STATE_MANIFEST_ED25519_PUBLIC_KEY_FILE"],
            "-rawin",
            "-in",
            payload_path,
            "-sigfile",
            signature_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if verify_result.returncode != 0:
      stderr = verify_result.stderr.decode("utf-8", errors="replace").strip()
      print(
          f"invalid lane-state manifest for {source}: signature_ed25519_base64 verification failed ({stderr or 'openssl verify error'})",
          file=os.sys.stderr,
      )
      raise SystemExit(1)
  finally:
    Path(payload_path).unlink(missing_ok=True)
    Path(signature_path).unlink(missing_ok=True)
else:
  print(f"invalid lane-state manifest signing mode: {expected_sign_mode}", file=os.sys.stderr)
  raise SystemExit(1)
PY
}

lane_state_load_file() {
  local lane_state_file="$1"
  local source_label="$2"
  if [[ ! -f "$lane_state_file" ]]; then
    return
  fi
  lane_state_verify_manifest "$lane_state_file" "$source_label"
  local line_no=0
  while IFS=$'\t' read -r lane_id suite mode total pass fail xfail xpass error skip updated_at_utc compat_policy_version config_hash extra; do
    line_no=$((line_no + 1))
    if [[ -z "$lane_id" ]]; then
      continue
    fi
    if [[ "$line_no" == "1" && "$lane_id" == "lane_id" ]]; then
      continue
    fi
    if [[ -z "$config_hash" && "$compat_policy_version" =~ ^[0-9a-f]{64}$ ]]; then
      # Backward compatibility for pre-policy rows with 12 fields where
      # config_hash occupied the final column.
      config_hash="$compat_policy_version"
      compat_policy_version="legacy"
    fi
    if [[ -n "$extra" ]]; then
      echo "invalid lane state row ${source_label}:${line_no}: expected 11, 12, or 13 tab-separated fields" >&2
      exit 1
    fi
    if [[ -z "$suite" || -z "$mode" || -z "$updated_at_utc" ]]; then
      echo "invalid lane state row ${source_label}:${line_no}: missing required fields" >&2
      exit 1
    fi
    for value in "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip"; do
      if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "invalid lane state row ${source_label}:${line_no}: non-integer counters" >&2
        exit 1
      fi
    done
    if ! [[ "$updated_at_utc" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
      echo "invalid lane state row ${source_label}:${line_no}: updated_at_utc must be UTC RFC3339 timestamp" >&2
      exit 1
    fi
    if [[ -n "$config_hash" && ! "$config_hash" =~ ^[0-9a-f]{64}$ ]]; then
      echo "invalid lane state row ${source_label}:${line_no}: invalid config_hash (expected 64 hex chars)" >&2
      exit 1
    fi
    if [[ -n "$compat_policy_version" && "$compat_policy_version" != "legacy" && ! "$compat_policy_version" =~ ^[0-9]+$ ]]; then
      echo "invalid lane state row ${source_label}:${line_no}: invalid compat_policy_version (expected integer or legacy)" >&2
      exit 1
    fi
    lane_state_merge_row "$lane_id" "$(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" \
      "$lane_id" "$suite" "$mode" "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip" "$updated_at_utc" "$compat_policy_version" "$config_hash")" "${source_label}:${line_no}"
  done < "$lane_state_file"
}

if [[ -n "$LANE_STATE_TSV" ]]; then
  LANE_STATE_ACTIVE_CONFIG_HASH="$(compute_lane_state_config_hash)"
  if [[ "$RESET_LANE_STATE" == "1" ]]; then
    lane_state_write_file
  else
    lane_state_load_file "$LANE_STATE_TSV" "$LANE_STATE_TSV"
  fi
  for merge_lane_state_file in "${MERGE_LANE_STATE_TSVS[@]}"; do
    lane_state_load_file "$merge_lane_state_file" "$merge_lane_state_file"
  done
  if [[ "${#MERGE_LANE_STATE_TSVS[@]}" -gt 0 ]]; then
    lane_state_write_file
  fi
fi

lane_resume_from_state() {
  local lane_id="$1"
  if [[ "$RESUME_FROM_LANE_STATE" != "1" ]]; then
    return 1
  fi
  local row="${LANE_STATE_ROW_BY_ID[$lane_id]-}"
  if [[ -z "$row" ]]; then
    return 1
  fi
  local saved_lane suite mode total pass fail xfail xpass error skip updated_at_utc compat_policy_version config_hash
  IFS=$'\t' read -r saved_lane suite mode total pass fail xfail xpass error skip updated_at_utc compat_policy_version config_hash <<< "$row"
  if [[ -z "$compat_policy_version" || "$compat_policy_version" == "legacy" ]]; then
    echo "lane state compatibility policy version missing for lane ${lane_id}; rerun with --reset-lane-state" >&2
    exit 1
  fi
  if [[ "$compat_policy_version" != "$LANE_STATE_COMPAT_POLICY_VERSION" ]]; then
    echo "lane state compatibility policy version mismatch for lane ${lane_id}; expected ${LANE_STATE_COMPAT_POLICY_VERSION}, found ${compat_policy_version}; rerun with --reset-lane-state" >&2
    exit 1
  fi
  if [[ -z "$config_hash" ]]; then
    echo "lane state config hash missing for lane ${lane_id}; rerun with --reset-lane-state" >&2
    exit 1
  fi
  if [[ "$config_hash" != "$LANE_STATE_ACTIVE_CONFIG_HASH" ]]; then
    echo "lane state config hash mismatch for lane ${lane_id}; expected ${LANE_STATE_ACTIVE_CONFIG_HASH}, found ${config_hash}; rerun with --reset-lane-state" >&2
    exit 1
  fi
  echo "==> ${lane_id} (resume from lane state @ ${updated_at_utc})" | tee -a "$OUT_DIR/resume.log"
  record_result "$suite" "$mode" "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip"
  return 0
}

run_suite() {
  local name="$1"; shift
  local log="$OUT_DIR/${name}.log"
  local ec=0
  echo "==> ${name}" | tee "$log"
  set +e
  "$@" >>"$log" 2>&1
  ec=$?
  set -e
  echo "$ec" > "$OUT_DIR/${name}.exit"
  return $ec
}

lane_enabled() {
  local lane_id="$1"
  if [[ -n "$INCLUDE_LANE_REGEX" ]]; then
    if ! printf '%s\n' "$lane_id" | grep -Eq "$INCLUDE_LANE_REGEX"; then
      return 1
    fi
  fi
  if [[ -n "$EXCLUDE_LANE_REGEX" ]]; then
    if printf '%s\n' "$lane_id" | grep -Eq "$EXCLUDE_LANE_REGEX"; then
      return 1
    fi
  fi
  return 0
}

extract_kv() {
  local line="$1"
  local key="$2"
  echo "$line" | tr ' ' '\n' | sed -n "s/^${key}=\([0-9]\+\)$/\\1/p"
}

summarize_opentitan_xprop_file() {
  local xprop_file="$1"
  if [[ ! -s "$xprop_file" ]]; then
    echo ""
    return 0
  fi
  XPROP_FILE="$xprop_file" python3 - <<'PY'
import os
import re
from collections import defaultdict
from pathlib import Path

path = Path(os.environ["XPROP_FILE"])
if not path.exists():
    print("")
    raise SystemExit(0)

def normalize(token: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")
    if not value:
        value = "unknown"
    return value

counts = defaultdict(int)
rows = 0
with path.open(encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        impl = normalize(parts[1].strip())
        status = normalize(parts[0].strip())
        diag = normalize(parts[3].strip())
        result = normalize(parts[4].strip())
        counters = parts[5].strip()
        rows += 1
        counts[f"xprop_impl_{impl}_cases"] += 1
        counts[f"xprop_impl_{impl}_status_{status}"] += 1
        counts[f"xprop_impl_{impl}_diag_{diag}"] += 1
        counts[f"xprop_impl_{impl}_result_{result}"] += 1
        counts[f"xprop_status_{status}"] += 1
        counts[f"xprop_diag_{diag}"] += 1
        counts[f"xprop_result_{result}"] += 1
        if counters:
            for item in counters.split(","):
                item = item.strip()
                if not item or "=" not in item:
                    continue
                key, raw_val = item.split("=", 1)
                key = normalize(key)
                try:
                    val = int(raw_val)
                except ValueError:
                    continue
                counts[f"xprop_impl_{impl}_counter_{key}"] += val
                counts[f"xprop_counter_{key}"] += val

if rows <= 0:
    print("")
    raise SystemExit(0)

counts["xprop_cases"] = rows
parts = [f"{key}={counts[key]}" for key in sorted(counts)]
print(" ".join(parts))
PY
}

summarize_bmc_case_file() {
  local case_file="$1"
  if [[ ! -s "$case_file" ]]; then
    echo ""
    return 0
  fi
  BMC_CASE_FILE="$case_file" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["BMC_CASE_FILE"])
if not path.exists():
    print("")
    raise SystemExit(0)

timeout_cases = 0
unknown_cases = 0
with path.open(encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        status = line.split("\t", 1)[0].strip().upper()
        if status == "TIMEOUT":
            timeout_cases += 1
        elif status == "UNKNOWN":
            unknown_cases += 1

print(f"bmc_timeout_cases={timeout_cases} bmc_unknown_cases={unknown_cases}")
PY
}

results_tsv="$OUT_DIR/summary.tsv"
: > "$results_tsv"

printf "suite\tmode\ttotal\tpass\tfail\txfail\txpass\terror\tskip\tsummary\n" >> "$results_tsv"

record_result_with_summary() {
  local suite="$1" mode="$2" total="$3" pass="$4" fail="$5" xfail="$6" xpass="$7" error="$8" skip="$9" summary="${10}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$suite" "$mode" "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip" "$summary" >> "$results_tsv"
  if [[ -n "$LANE_STATE_TSV" ]]; then
    local lane_id="${suite}/${mode}"
    local updated_at_utc
    updated_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    lane_state_register_row "$lane_id" "$(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" \
      "$lane_id" "$suite" "$mode" "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip" "$updated_at_utc" "$LANE_STATE_COMPAT_POLICY_VERSION" "$LANE_STATE_ACTIVE_CONFIG_HASH")" "current-run"
    lane_state_write_file
  fi
}

record_result() {
  local suite="$1" mode="$2" total="$3" pass="$4" fail="$5" xfail="$6" xpass="$7" error="$8" skip="$9"
  local summary="total=${total} pass=${pass} fail=${fail} xfail=${xfail} xpass=${xpass} error=${error} skip=${skip}"
  record_result_with_summary "$suite" "$mode" "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip" "$summary"
}

record_simple_result() {
  local suite="$1" mode="$2" exit_code="$3"
  local total=1 pass=0 fail=0 xfail=0 xpass=0 error=0 skip=0
  if [[ "$exit_code" == "0" ]]; then
    pass=1
  else
    fail=1
  fi
  record_result "$suite" "$mode" "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip"
}

# sv-tests BMC
if [[ -d "$SV_TESTS_DIR" ]] && lane_enabled "sv-tests/BMC"; then
  if lane_resume_from_state "sv-tests/BMC"; then
    :
  else
    run_suite sv-tests-bmc \
      env OUT="$OUT_DIR/sv-tests-bmc-results.txt" \
      BMC_RUN_SMTLIB="$BMC_RUN_SMTLIB" \
      ALLOW_MULTI_CLOCK="$BMC_ALLOW_MULTI_CLOCK" \
      BMC_ASSUME_KNOWN_INPUTS="$BMC_ASSUME_KNOWN_INPUTS" \
      Z3_BIN="$Z3_BIN" \
      utils/run_sv_tests_circt_bmc.sh "$SV_TESTS_DIR" || true
    line=$(grep -E "sv-tests SVA summary:" "$OUT_DIR/sv-tests-bmc.log" | tail -1 || true)
    if [[ -n "$line" ]]; then
      total=$(extract_kv "$line" total)
      pass=$(extract_kv "$line" pass)
      fail=$(extract_kv "$line" fail)
      xfail=$(extract_kv "$line" xfail)
      xpass=$(extract_kv "$line" xpass)
      error=$(extract_kv "$line" error)
      skip=$(extract_kv "$line" skip)
      summary="total=${total} pass=${pass} fail=${fail} xfail=${xfail} xpass=${xpass} error=${error} skip=${skip}"
      bmc_case_summary="$(summarize_bmc_case_file "$OUT_DIR/sv-tests-bmc-results.txt")"
      if [[ -n "$bmc_case_summary" ]]; then
        summary="${summary} ${bmc_case_summary}"
      fi
      record_result_with_summary "sv-tests" "BMC" "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip" "$summary"
    fi
  fi
fi

# sv-tests LEC
if [[ -d "$SV_TESTS_DIR" ]] && lane_enabled "sv-tests/LEC"; then
  if lane_resume_from_state "sv-tests/LEC"; then
    :
  else
    run_suite sv-tests-lec \
      env OUT="$OUT_DIR/sv-tests-lec-results.txt" \
      LEC_ASSUME_KNOWN_INPUTS="$LEC_ASSUME_KNOWN_INPUTS" \
      LEC_ACCEPT_XPROP_ONLY="$LEC_ACCEPT_XPROP_ONLY" \
      Z3_BIN="$Z3_BIN" \
      utils/run_sv_tests_circt_lec.sh "$SV_TESTS_DIR" || true
    line=$(grep -E "sv-tests LEC summary:" "$OUT_DIR/sv-tests-lec.log" | tail -1 || true)
    if [[ -n "$line" ]]; then
      total=$(extract_kv "$line" total)
      pass=$(extract_kv "$line" pass)
      fail=$(extract_kv "$line" fail)
      error=$(extract_kv "$line" error)
      skip=$(extract_kv "$line" skip)
      record_result "sv-tests" "LEC" "$total" "$pass" "$fail" 0 0 "$error" "$skip"
    fi
  fi
fi

# verilator-verification BMC
if [[ -d "$VERILATOR_DIR" ]] && lane_enabled "verilator-verification/BMC"; then
  if lane_resume_from_state "verilator-verification/BMC"; then
    :
  else
    run_suite verilator-bmc \
      env OUT="$OUT_DIR/verilator-bmc-results.txt" \
      BMC_RUN_SMTLIB="$BMC_RUN_SMTLIB" \
      ALLOW_MULTI_CLOCK="$BMC_ALLOW_MULTI_CLOCK" \
      BMC_ASSUME_KNOWN_INPUTS="$BMC_ASSUME_KNOWN_INPUTS" \
      Z3_BIN="$Z3_BIN" \
      utils/run_verilator_verification_circt_bmc.sh "$VERILATOR_DIR" || true
    line=$(grep -E "verilator-verification summary:" "$OUT_DIR/verilator-bmc.log" | tail -1 || true)
    if [[ -n "$line" ]]; then
      total=$(extract_kv "$line" total)
      pass=$(extract_kv "$line" pass)
      fail=$(extract_kv "$line" fail)
      xfail=$(extract_kv "$line" xfail)
      xpass=$(extract_kv "$line" xpass)
      error=$(extract_kv "$line" error)
      skip=$(extract_kv "$line" skip)
      summary="total=${total} pass=${pass} fail=${fail} xfail=${xfail} xpass=${xpass} error=${error} skip=${skip}"
      bmc_case_summary="$(summarize_bmc_case_file "$OUT_DIR/verilator-bmc-results.txt")"
      if [[ -n "$bmc_case_summary" ]]; then
        summary="${summary} ${bmc_case_summary}"
      fi
      record_result_with_summary "verilator-verification" "BMC" "$total" "$pass" "$fail" "$xfail" "$xpass" "$error" "$skip" "$summary"
    fi
  fi
fi

# verilator-verification LEC
if [[ -d "$VERILATOR_DIR" ]] && lane_enabled "verilator-verification/LEC"; then
  if lane_resume_from_state "verilator-verification/LEC"; then
    :
  else
    run_suite verilator-lec \
      env OUT="$OUT_DIR/verilator-lec-results.txt" \
      LEC_ASSUME_KNOWN_INPUTS="$LEC_ASSUME_KNOWN_INPUTS" \
      LEC_ACCEPT_XPROP_ONLY="$LEC_ACCEPT_XPROP_ONLY" \
      Z3_BIN="$Z3_BIN" \
      utils/run_verilator_verification_circt_lec.sh "$VERILATOR_DIR" || true
    line=$(grep -E "verilator-verification LEC summary:" "$OUT_DIR/verilator-lec.log" | tail -1 || true)
    if [[ -n "$line" ]]; then
      total=$(extract_kv "$line" total)
      pass=$(extract_kv "$line" pass)
      fail=$(extract_kv "$line" fail)
      error=$(extract_kv "$line" error)
      skip=$(extract_kv "$line" skip)
      record_result "verilator-verification" "LEC" "$total" "$pass" "$fail" 0 0 "$error" "$skip"
    fi
  fi
fi

# yosys SVA BMC
if [[ -d "$YOSYS_DIR" ]] && lane_enabled "yosys/tests/sva/BMC"; then
  if lane_resume_from_state "yosys/tests/sva/BMC"; then
    :
  else
    # NOTE: Do not pass BMC_ASSUME_KNOWN_INPUTS here; the yosys script defaults
    # it to 1 because yosys SVA tests are 2-state and need --assume-known-inputs
    # to avoid spurious X-driven counterexamples.  Only forward an explicit
    # override from the user (--bmc-assume-known-inputs flag).
    yosys_bmc_env=(OUT="$OUT_DIR/yosys-bmc-results.txt"
      BMC_RUN_SMTLIB="$BMC_RUN_SMTLIB"
      ALLOW_MULTI_CLOCK="$BMC_ALLOW_MULTI_CLOCK"
      Z3_BIN="$Z3_BIN")
    if [[ "$BMC_ASSUME_KNOWN_INPUTS" == "1" ]]; then
      yosys_bmc_env+=(BMC_ASSUME_KNOWN_INPUTS=1)
    fi
    run_suite yosys-bmc \
      env "${yosys_bmc_env[@]}" \
      utils/run_yosys_sva_circt_bmc.sh "$YOSYS_DIR" || true
    line=$(grep -E "yosys SVA summary:" "$OUT_DIR/yosys-bmc.log" | tail -1 || true)
    if [[ -n "$line" ]]; then
      total=$(echo "$line" | sed -n 's/.*summary: \([0-9]\+\) tests.*/\1/p')
      failures=$(echo "$line" | sed -n 's/.*failures=\([0-9]\+\).*/\1/p')
      skipped=$(echo "$line" | sed -n 's/.*skipped=\([0-9]\+\).*/\1/p')
      pass=$((total - failures - skipped))
      record_result "yosys/tests/sva" "BMC" "$total" "$pass" "$failures" 0 0 0 "$skipped"
    fi
  fi
fi

# yosys SVA LEC
if [[ -d "$YOSYS_DIR" ]] && lane_enabled "yosys/tests/sva/LEC"; then
  if lane_resume_from_state "yosys/tests/sva/LEC"; then
    :
  else
    run_suite yosys-lec \
      env OUT="$OUT_DIR/yosys-lec-results.txt" \
      LEC_ASSUME_KNOWN_INPUTS="$LEC_ASSUME_KNOWN_INPUTS" \
      LEC_ACCEPT_XPROP_ONLY="$LEC_ACCEPT_XPROP_ONLY" \
      Z3_BIN="$Z3_BIN" \
      utils/run_yosys_sva_circt_lec.sh "$YOSYS_DIR" || true
    line=$(grep -E "yosys LEC summary:" "$OUT_DIR/yosys-lec.log" | tail -1 || true)
    if [[ -n "$line" ]]; then
      total=$(extract_kv "$line" total)
      pass=$(extract_kv "$line" pass)
      fail=$(extract_kv "$line" fail)
      error=$(extract_kv "$line" error)
      skip=$(extract_kv "$line" skip)
      record_result "yosys/tests/sva" "LEC" "$total" "$pass" "$fail" 0 0 "$error" "$skip"
    fi
  fi
fi

run_opentitan_lec_lane() {
  local lane_id="$1"
  local mode_name="$2"
  local suite_name="$3"
  local case_results="$4"
  local xprop_summary_file="$5"
  local workdir="$6"
  local strict_x="$7"

  if ! lane_enabled "$lane_id"; then
    return
  fi
  if lane_resume_from_state "$lane_id"; then
    return
  fi

  : > "$case_results"
  : > "$xprop_summary_file"
  rm -rf "$workdir"

  local opentitan_lec_args=(--opentitan-root "$OPENTITAN_DIR")
  if [[ -n "$OPENTITAN_LEC_IMPL_FILTER" ]]; then
    opentitan_lec_args+=(--impl-filter "$OPENTITAN_LEC_IMPL_FILTER")
  fi
  if [[ "$OPENTITAN_LEC_INCLUDE_MASKED" == "1" ]]; then
    opentitan_lec_args+=(--include-masked)
  fi

  local opentitan_lec_env=(OUT="$case_results"
    OUT_XPROP_SUMMARY="$xprop_summary_file"
    CIRCT_VERILOG="$CIRCT_VERILOG_BIN_OPENTITAN")
  if [[ "$strict_x" == "1" ]]; then
    opentitan_lec_env+=(LEC_X_OPTIMISTIC=0 LEC_MODE_LABEL=LEC_STRICT)
    if [[ "$OPENTITAN_LEC_STRICT_DUMP_UNKNOWN_SOURCES" == "1" ]]; then
      opentitan_lec_env+=(LEC_DUMP_UNKNOWN_SOURCES=1)
    fi
  fi
  if [[ "$LEC_ASSUME_KNOWN_INPUTS" == "1" ]]; then
    opentitan_lec_env+=(LEC_ASSUME_KNOWN_INPUTS=1)
  fi

  run_suite "$suite_name" \
    env "${opentitan_lec_env[@]}" \
    utils/run_opentitan_circt_lec.py --workdir "$workdir" "${opentitan_lec_args[@]}" || true

  if [[ ! -s "$case_results" ]]; then
    printf "FAIL\taes_sbox\tmissing_results\topentitan\t%s\n" "$mode_name" > "$case_results"
    local missing_summary="total=1 pass=0 fail=1 xfail=0 xpass=0 error=1 skip=0 missing_results=1"
    record_result_with_summary "opentitan" "$mode_name" 1 0 1 0 0 1 0 "$missing_summary"
    return
  fi

  local counts total pass fail
  counts="$(
    OPENTITAN_CASE_RESULTS_FILE="$case_results" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["OPENTITAN_CASE_RESULTS_FILE"])
total = 0
passed = 0
failed = 0
for line in path.read_text().splitlines():
  cols = line.split("\t")
  if not cols or not cols[0]:
    continue
  total += 1
  if cols[0].strip().upper() == "PASS":
    passed += 1
  else:
    failed += 1
print(f"{total}\t{passed}\t{failed}")
PY
  )"
  IFS=$'\t' read -r total pass fail <<< "$counts"

  local summary="total=${total} pass=${pass} fail=${fail} xfail=0 xpass=0 error=0 skip=0"
  local xprop_summary
  xprop_summary="$(summarize_opentitan_xprop_file "$xprop_summary_file")"
  if [[ -n "$xprop_summary" ]]; then
    summary="${summary} ${xprop_summary}"
  fi
  record_result_with_summary "opentitan" "$mode_name" "$total" "$pass" "$fail" 0 0 0 0 "$summary"
}

# OpenTitan LEC (optional)
if [[ "$WITH_OPENTITAN" == "1" ]]; then
  run_opentitan_lec_lane \
    "opentitan/LEC" \
    "LEC" \
    "opentitan-lec" \
    "$OUT_DIR/opentitan-lec-results.txt" \
    "$OUT_DIR/opentitan-lec-xprop-summary.tsv" \
    "$OUT_DIR/opentitan-lec-work" \
    "0"
fi

# OpenTitan strict LEC audit lane (optional)
if [[ "$WITH_OPENTITAN_LEC_STRICT" == "1" ]]; then
  run_opentitan_lec_lane \
    "opentitan/LEC_STRICT" \
    "LEC_STRICT" \
    "opentitan-lec-strict" \
    "$OUT_DIR/opentitan-lec-strict-results.txt" \
    "$OUT_DIR/opentitan-lec-strict-xprop-summary.tsv" \
    "$OUT_DIR/opentitan-lec-strict-work" \
    "1"
fi

run_opentitan_e2e_lane() {
  local lane_id="$1"
  local mode_name="$2"
  local suite_name="$3"
  local lane_lec_x_mode="$4"
  local lane_out_dir="$5"
  local lane_results_tsv="$6"
  local case_results="$7"

  if ! lane_enabled "$lane_id"; then
    return
  fi
  if lane_resume_from_state "$lane_id"; then
    return
  fi

  : > "$case_results"
  local opentitan_e2e_cmd=(
    utils/run_opentitan_formal_e2e.sh
    --opentitan-root "$OPENTITAN_DIR"
    --out-dir "$lane_out_dir"
    --results-file "$lane_results_tsv"
  )
  if [[ "$lane_lec_x_mode" == "strict" ]]; then
    opentitan_e2e_cmd+=(--lec-strict-x)
  else
    opentitan_e2e_cmd+=(--lec-x-optimistic)
  fi
  if [[ -n "$OPENTITAN_E2E_SIM_TARGETS" ]]; then
    opentitan_e2e_cmd+=(--sim-targets "$OPENTITAN_E2E_SIM_TARGETS")
  fi
  if [[ -n "$OPENTITAN_E2E_VERILOG_TARGETS" ]]; then
    opentitan_e2e_cmd+=(--verilog-targets "$OPENTITAN_E2E_VERILOG_TARGETS")
  fi
  if [[ -n "$OPENTITAN_E2E_SIM_TIMEOUT" ]]; then
    opentitan_e2e_cmd+=(--sim-timeout "$OPENTITAN_E2E_SIM_TIMEOUT")
  fi
  if [[ -n "$OPENTITAN_E2E_IMPL_FILTER" ]]; then
    opentitan_e2e_cmd+=(--impl-filter "$OPENTITAN_E2E_IMPL_FILTER")
  fi
  if [[ "$OPENTITAN_E2E_INCLUDE_MASKED" == "1" ]]; then
    opentitan_e2e_cmd+=(--include-masked)
  fi
  if [[ "$LEC_ASSUME_KNOWN_INPUTS" == "1" ]]; then
    opentitan_e2e_cmd+=(--lec-assume-known-inputs)
  fi
  run_suite "$suite_name" \
    env CIRCT_VERILOG="$CIRCT_VERILOG_BIN_OPENTITAN" \
    "${opentitan_e2e_cmd[@]}" || true

  local opentitan_e2e_total=""
  local opentitan_e2e_pass=""
  local opentitan_e2e_fail=""
  if [[ -s "$lane_results_tsv" ]]; then
    opentitan_e2e_counts="$(OPENTITAN_E2E_RESULTS_TSV="$lane_results_tsv" OPENTITAN_E2E_CASE_RESULTS="$case_results" OPENTITAN_E2E_MODE_NAME="$mode_name" python3 - <<'PY'
import csv
import os
from pathlib import Path

results_path = Path(os.environ["OPENTITAN_E2E_RESULTS_TSV"])
case_results_path = Path(os.environ["OPENTITAN_E2E_CASE_RESULTS"])
mode_name = os.environ["OPENTITAN_E2E_MODE_NAME"]
rows = []
total = 0
passed = 0
failed = 0
fail_like_statuses = {"FAIL", "ERROR", "XFAIL", "XPASS", "EFAIL", "TIMEOUT", "UNKNOWN"}
with results_path.open() as f:
  reader = csv.DictReader(f, delimiter="\t")
  for row in reader:
    kind = (row.get("kind") or "").strip()
    target = (row.get("target") or "").strip()
    status = (row.get("status") or "").strip().upper()
    detail = (row.get("detail") or "").strip()
    artifact = (row.get("artifact") or "").strip()
    if not kind and not target:
      continue
    total += 1
    if status == "PASS":
      case_status = "PASS"
    elif status in fail_like_statuses:
      case_status = status
    else:
      case_status = "FAIL"
    if case_status == "PASS":
      passed += 1
    else:
      failed += 1
    base = f"{kind}:{target}" if kind and target else (kind or target)
    path = artifact or detail or base
    rows.append((case_status, base, path, "opentitan", mode_name))

rows.sort(key=lambda r: (r[1], r[0], r[2]))
with case_results_path.open("w") as f:
  for row in rows:
    f.write("\t".join(row) + "\n")

print(f"{total}\t{passed}\t{failed}")
PY
)"
    IFS=$'\t' read -r opentitan_e2e_total opentitan_e2e_pass opentitan_e2e_fail <<< "$opentitan_e2e_counts"
  fi
  if [[ -z "$opentitan_e2e_total" || -z "$opentitan_e2e_pass" || -z "$opentitan_e2e_fail" ]]; then
    local line
    line=$(grep -E "OpenTitan E2E summary: pass=[0-9]+ fail=[0-9]+" "$OUT_DIR/${suite_name}.log" | tail -1 || true)
    if [[ -n "$line" ]]; then
      opentitan_e2e_pass=$(extract_kv "$line" pass)
      opentitan_e2e_fail=$(extract_kv "$line" fail)
      if [[ -n "$opentitan_e2e_pass" && -n "$opentitan_e2e_fail" ]]; then
        opentitan_e2e_total=$((opentitan_e2e_pass + opentitan_e2e_fail))
      fi
    fi
  fi
  if [[ -n "$opentitan_e2e_total" && -n "$opentitan_e2e_pass" && -n "$opentitan_e2e_fail" ]]; then
    record_result "opentitan" "$mode_name" "$opentitan_e2e_total" "$opentitan_e2e_pass" "$opentitan_e2e_fail" 0 0 0 0
  fi
}

# OpenTitan non-smoke end-to-end parity lane (optional)
if [[ "$WITH_OPENTITAN_E2E" == "1" ]]; then
  run_opentitan_e2e_lane \
    "opentitan/E2E" \
    "E2E" \
    "opentitan-e2e" \
    "$OPENTITAN_E2E_LEC_X_MODE" \
    "$OUT_DIR/opentitan-formal-e2e" \
    "$OUT_DIR/opentitan-e2e-results.tsv" \
    "$OUT_DIR/opentitan-e2e-results.txt"
fi

# OpenTitan strict non-smoke end-to-end audit lane (optional)
if [[ "$WITH_OPENTITAN_E2E_STRICT" == "1" ]]; then
  run_opentitan_e2e_lane \
    "opentitan/E2E_STRICT" \
    "E2E_STRICT" \
    "opentitan-e2e-strict" \
    "strict" \
    "$OUT_DIR/opentitan-formal-e2e-strict" \
    "$OUT_DIR/opentitan-e2e-strict-results.tsv" \
    "$OUT_DIR/opentitan-e2e-strict-results.txt"
fi

# OpenTitan E2E mode-diff synthesis (default vs strict)
if [[ "$WITH_OPENTITAN_E2E" == "1" && "$WITH_OPENTITAN_E2E_STRICT" == "1" ]]; then
  opentitan_e2e_default_case_results="$OUT_DIR/opentitan-e2e-results.txt"
  opentitan_e2e_strict_case_results="$OUT_DIR/opentitan-e2e-strict-results.txt"
  opentitan_e2e_mode_diff_tsv="$OUT_DIR/opentitan-e2e-mode-diff.tsv"
  opentitan_e2e_mode_diff_results="$OUT_DIR/opentitan-e2e-mode-diff-results.txt"
  opentitan_e2e_mode_diff_metrics_tsv="$OUT_DIR/opentitan-e2e-mode-diff-metrics.tsv"
  if [[ -s "$opentitan_e2e_default_case_results" && -s "$opentitan_e2e_strict_case_results" ]]; then
    opentitan_e2e_mode_diff_counts="$(
      OPENTITAN_E2E_DEFAULT_CASE_RESULTS="$opentitan_e2e_default_case_results" \
      OPENTITAN_E2E_STRICT_CASE_RESULTS="$opentitan_e2e_strict_case_results" \
      OPENTITAN_E2E_MODE_DIFF_TSV="$opentitan_e2e_mode_diff_tsv" \
      OPENTITAN_E2E_MODE_DIFF_RESULTS="$opentitan_e2e_mode_diff_results" \
      OPENTITAN_E2E_MODE_DIFF_METRICS_TSV="$opentitan_e2e_mode_diff_metrics_tsv" \
      python3 - <<'PY'
import csv
import os
from pathlib import Path

default_path = Path(os.environ["OPENTITAN_E2E_DEFAULT_CASE_RESULTS"])
strict_path = Path(os.environ["OPENTITAN_E2E_STRICT_CASE_RESULTS"])
diff_tsv_path = Path(os.environ["OPENTITAN_E2E_MODE_DIFF_TSV"])
diff_results_path = Path(os.environ["OPENTITAN_E2E_MODE_DIFF_RESULTS"])
metrics_tsv_path = Path(os.environ["OPENTITAN_E2E_MODE_DIFF_METRICS_TSV"])


def load_rows(path: Path):
  rows = {}
  with path.open() as f:
    for line in f:
      line = line.rstrip("\n")
      if not line:
        continue
      parts = line.split("\t")
      if len(parts) < 3:
        continue
      status = parts[0].strip().upper()
      base = parts[1].strip()
      detail = parts[2].strip()
      if not base:
        continue
      rows[base] = {
          "status": status or "UNKNOWN",
          "detail": detail,
      }
  return rows


default_rows = load_rows(default_path)
strict_rows = load_rows(strict_path)
bases = sorted(set(default_rows.keys()) | set(strict_rows.keys()))
classification_keys = [
    "same_status",
    "strict_only_fail",
    "strict_only_pass",
    "status_diff",
    "missing_in_e2e",
    "missing_in_e2e_strict",
]
classification_counts = {k: 0 for k in classification_keys}

with diff_tsv_path.open("w", newline="") as f:
  writer = csv.writer(f, delimiter="\t")
  writer.writerow(
      [
          "status",
          "base",
          "classification",
          "status_e2e",
          "status_e2e_strict",
          "detail_e2e",
          "detail_e2e_strict",
      ]
  )
  fail_count = 0
  for base in bases:
    e2e = default_rows.get(base, {"status": "MISSING", "detail": ""})
    strict = strict_rows.get(base, {"status": "MISSING", "detail": ""})
    status_e2e = e2e["status"]
    status_strict = strict["status"]
    detail_e2e = e2e["detail"]
    detail_strict = strict["detail"]
    if status_e2e == status_strict and status_e2e != "MISSING":
      classification = "same_status"
      status = "PASS"
    elif status_e2e == "MISSING":
      classification = "missing_in_e2e"
      status = "FAIL"
    elif status_strict == "MISSING":
      classification = "missing_in_e2e_strict"
      status = "FAIL"
    elif status_e2e == "PASS" and status_strict != "PASS":
      classification = "strict_only_fail"
      status = "FAIL"
    elif status_e2e != "PASS" and status_strict == "PASS":
      classification = "strict_only_pass"
      status = "FAIL"
    else:
      classification = "status_diff"
      status = "FAIL"
    classification_counts[classification] += 1
    writer.writerow(
        [
            status,
            base,
            classification,
            status_e2e,
            status_strict,
            detail_e2e,
            detail_strict,
        ]
    )
    if status == "FAIL":
      fail_count += 1

with diff_results_path.open("w") as f:
  with diff_tsv_path.open() as src:
    reader = csv.DictReader(src, delimiter="\t")
    for row in reader:
      if row.get("status", "").upper() != "FAIL":
        continue
      base = (row.get("base") or "").strip()
      classification = (row.get("classification") or "").strip()
      status_e2e = (row.get("status_e2e") or "").strip()
      status_strict = (row.get("status_e2e_strict") or "").strip()
      detail = (
          f"{classification};e2e={status_e2e};e2e_strict={status_strict}"
      )
      f.write(f"FAIL\t{base}\t{detail}\topentitan\tE2E_MODE_DIFF\n")

with metrics_tsv_path.open("w", newline="") as f:
  writer = csv.writer(f, delimiter="\t")
  writer.writerow(["metric", "value"])
  writer.writerow(["total_cases", len(bases)])
  writer.writerow(["pass_cases", len(bases) - fail_count])
  writer.writerow(["fail_cases", fail_count])
  for key in classification_keys:
    writer.writerow([key, classification_counts[key]])

print(
    "\t".join(
        [
            str(len(bases)),
            str(len(bases) - fail_count),
            str(fail_count),
            str(classification_counts["same_status"]),
            str(classification_counts["strict_only_fail"]),
            str(classification_counts["strict_only_pass"]),
            str(classification_counts["status_diff"]),
            str(classification_counts["missing_in_e2e"]),
            str(classification_counts["missing_in_e2e_strict"]),
        ]
    )
)
PY
)"
    IFS=$'\t' read -r \
      opentitan_e2e_mode_diff_total \
      opentitan_e2e_mode_diff_pass \
      opentitan_e2e_mode_diff_fail \
      opentitan_e2e_mode_diff_same_status \
      opentitan_e2e_mode_diff_strict_only_fail \
      opentitan_e2e_mode_diff_strict_only_pass \
      opentitan_e2e_mode_diff_status_diff \
      opentitan_e2e_mode_diff_missing_in_e2e \
      opentitan_e2e_mode_diff_missing_in_e2e_strict <<< "$opentitan_e2e_mode_diff_counts"
    if [[ -n "$opentitan_e2e_mode_diff_total" && -n "$opentitan_e2e_mode_diff_pass" && -n "$opentitan_e2e_mode_diff_fail" ]]; then
      opentitan_e2e_mode_diff_summary="total=${opentitan_e2e_mode_diff_total} pass=${opentitan_e2e_mode_diff_pass} fail=${opentitan_e2e_mode_diff_fail} xfail=0 xpass=0 error=0 skip=0 same_status=${opentitan_e2e_mode_diff_same_status:-0} strict_only_fail=${opentitan_e2e_mode_diff_strict_only_fail:-0} strict_only_pass=${opentitan_e2e_mode_diff_strict_only_pass:-0} status_diff=${opentitan_e2e_mode_diff_status_diff:-0} missing_in_e2e=${opentitan_e2e_mode_diff_missing_in_e2e:-0} missing_in_e2e_strict=${opentitan_e2e_mode_diff_missing_in_e2e_strict:-0}"
      record_result_with_summary "opentitan" "E2E_MODE_DIFF" \
        "$opentitan_e2e_mode_diff_total" \
        "$opentitan_e2e_mode_diff_pass" \
        "$opentitan_e2e_mode_diff_fail" 0 0 0 0 \
        "$opentitan_e2e_mode_diff_summary"
    fi
  fi
fi

# AVIP compile smoke (optional)
if [[ "$WITH_AVIP" == "1" ]]; then
  avip_case_results="$OUT_DIR/avip-results.txt"
  : > "$avip_case_results"
  for avip in $AVIP_GLOB; do
    if [[ -d "$avip" ]]; then
      avip_name="$(basename "$avip")"
      avip_lane_id="avip/${avip_name}/compile"
      if ! lane_enabled "$avip_lane_id"; then
        continue
      fi
      if lane_resume_from_state "$avip_lane_id"; then
        continue
      fi
      run_suite "avip-${avip_name}" \
        env OUT="$OUT_DIR/${avip_name}-circt-verilog.log" \
        CIRCT_VERILOG="$CIRCT_VERILOG_BIN_AVIP" \
        utils/run_avip_circt_verilog.sh "$avip" || true
      if [[ -f "$OUT_DIR/avip-${avip_name}.exit" ]]; then
        avip_ec=$(cat "$OUT_DIR/avip-${avip_name}.exit")
        record_simple_result "avip/${avip_name}" "compile" "$avip_ec"
        avip_status="FAIL"
        if [[ "$avip_ec" == "0" ]]; then
          avip_status="PASS"
        fi
        printf "%s\t%s\t%s\t%s\t%s\n" \
          "$avip_status" "$avip_name" "$avip" "avip/${avip_name}" "compile" >> "$avip_case_results"
      fi
    fi
  done
  sort -o "$avip_case_results" "$avip_case_results"
fi

summary_txt="$OUT_DIR/summary.txt"
{
  echo "Formal suite summary (${DATE_STR})"
  printf "%-28s %-6s %-6s %s\n" "Suite" "Mode" "Status" "Details"
  echo "---------------------------------------------------------------"
  tail -n +2 "$results_tsv" | while IFS=$'\t' read -r suite mode total pass fail xfail xpass error skip summary; do
    status="PASS"
    if [[ "$mode" == "BMC" ]]; then
      if [[ "$fail" != "0" || "$error" != "0" || "$xpass" != "0" ]]; then
        status="FAIL"
      fi
    else
      if [[ "$fail" != "0" || "$error" != "0" ]]; then
        status="FAIL"
      fi
    fi
    printf "%-28s %-6s %-6s %s\n" "$suite" "$mode" "$status" "$summary"
  done
  echo "Logs: $OUT_DIR"
} | tee "$summary_txt"

OUT_DIR="$OUT_DIR" JSON_SUMMARY_FILE="$JSON_SUMMARY_FILE" DATE_STR="$DATE_STR" python3 - <<'PY'
import csv
import json
import os
import subprocess
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
summary_path = out_dir / "summary.tsv"
json_summary_path = Path(os.environ["JSON_SUMMARY_FILE"])

try:
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
except Exception:
    git_sha = ""

rows = []
with summary_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        total = int(row["total"])
        passed = int(row["pass"])
        xfail = int(row["xfail"])
        skipped = int(row["skip"])
        eligible = max(total - skipped, 0)
        pass_rate = 0.0
        if eligible > 0:
            pass_rate = ((passed + xfail) * 100.0) / eligible
        rows.append(
            {
                "suite": row["suite"],
                "mode": row["mode"],
                "total": total,
                "pass": passed,
                "fail": int(row["fail"]),
                "xfail": xfail,
                "xpass": int(row["xpass"]),
                "error": int(row["error"]),
                "skip": skipped,
                "pass_rate": round(pass_rate, 3),
                "summary": row["summary"],
            }
        )

payload = {
    "date": os.environ.get("DATE_STR", ""),
    "git_sha": git_sha,
    "rows": rows,
}
json_summary_path.parent.mkdir(parents=True, exist_ok=True)
json_summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

if [[ -n "$EXPECTED_FAILURES_FILE" || \
      "$FAIL_ON_UNEXPECTED_FAILURES" == "1" || \
      "$FAIL_ON_UNUSED_EXPECTED_FAILURES" == "1" || \
      -n "$PRUNE_EXPECTED_FAILURES_FILE" ]]; then
  OUT_DIR="$OUT_DIR" \
  JSON_SUMMARY_FILE="$JSON_SUMMARY_FILE" \
  EXPECTED_FAILURES_FILE="$EXPECTED_FAILURES_FILE" \
  FAIL_ON_UNEXPECTED_FAILURES="$FAIL_ON_UNEXPECTED_FAILURES" \
  FAIL_ON_UNUSED_EXPECTED_FAILURES="$FAIL_ON_UNUSED_EXPECTED_FAILURES" \
  EXPECTATIONS_DRY_RUN="$EXPECTATIONS_DRY_RUN" \
  EXPECTATIONS_DRY_RUN_RUN_ID="$RUN_ID" \
  EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS="$EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS" \
  EXPECTATIONS_DRY_RUN_REPORT_JSONL="$EXPECTATIONS_DRY_RUN_REPORT_JSONL" \
  PRUNE_EXPECTED_FAILURES_FILE="$PRUNE_EXPECTED_FAILURES_FILE" \
  PRUNE_EXPECTED_FAILURES_DROP_UNUSED="$PRUNE_EXPECTED_FAILURES_DROP_UNUSED" \
  python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
summary_path = out_dir / "summary.tsv"
json_summary_path = Path(os.environ["JSON_SUMMARY_FILE"])
expected_file_raw = os.environ.get("EXPECTED_FAILURES_FILE", "")
fail_on_unexpected = os.environ.get("FAIL_ON_UNEXPECTED_FAILURES", "0") == "1"
fail_on_unused = os.environ.get("FAIL_ON_UNUSED_EXPECTED_FAILURES", "0") == "1"
expectations_dry_run = os.environ.get("EXPECTATIONS_DRY_RUN", "0") == "1"
dry_run_run_id = os.environ.get("EXPECTATIONS_DRY_RUN_RUN_ID", "")
max_sample_rows = int(
    os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS", "5")
)
dry_run_report_jsonl_raw = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_JSONL", "")
prune_file_raw = os.environ.get("PRUNE_EXPECTED_FAILURES_FILE", "")
prune_drop_unused = (
    os.environ.get("PRUNE_EXPECTED_FAILURES_DROP_UNUSED", "0") == "1"
)
expected_path = Path(expected_file_raw) if expected_file_raw else None
prune_path = Path(prune_file_raw) if prune_file_raw else None
dry_run_report_jsonl_path = Path(dry_run_report_jsonl_raw) if dry_run_report_jsonl_raw else None
expected_summary_path = out_dir / "expected-failures-summary.tsv"

def emit_dry_run_report(payload):
  if dry_run_report_jsonl_path is None:
    return
  payload = dict(payload)
  payload.setdefault("run_id", dry_run_run_id)
  dry_run_report_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
  with dry_run_report_jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")

def sample_rows(rows):
  if max_sample_rows <= 0:
    return []
  return rows[:max_sample_rows]

required_cols = {"suite", "mode", "expected_fail", "expected_error"}
expected = {}
if expected_path is not None:
  with expected_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    fieldnames = set(reader.fieldnames or [])
    if not required_cols.issubset(fieldnames):
      missing = sorted(required_cols - fieldnames)
      raise SystemExit(
          f"invalid expected-failures file: missing required columns: {', '.join(missing)}"
      )
    for row in reader:
      suite = row.get("suite", "").strip()
      mode = row.get("mode", "").strip()
      if not suite or not mode:
        raise SystemExit("invalid expected-failures row: suite/mode must be non-empty")
      key = (suite, mode)
      if key in expected:
        raise SystemExit(
            f"invalid expected-failures file: duplicate row for suite={suite} mode={mode}"
        )
      try:
        expected_fail = int(row.get("expected_fail", "0"))
        expected_error = int(row.get("expected_error", "0"))
      except ValueError:
        raise SystemExit(
            f"invalid expected-failures row for suite={suite} mode={mode}: "
            "expected_fail/expected_error must be integers"
        )
      if expected_fail < 0 or expected_error < 0:
        raise SystemExit(
            f"invalid expected-failures row for suite={suite} mode={mode}: "
            "expected_fail/expected_error must be non-negative"
        )
      expected[key] = {
          "expected_fail": expected_fail,
          "expected_error": expected_error,
          "notes": row.get("notes", ""),
      }

rows = []
seen = set()
totals = {
    "actual_fail": 0,
    "actual_error": 0,
    "expected_fail": 0,
    "expected_error": 0,
    "unexpected_fail": 0,
    "unexpected_error": 0,
}

with summary_path.open() as f:
  reader = csv.DictReader(f, delimiter='\t')
  for row in reader:
    suite = row["suite"]
    mode = row["mode"]
    key = (suite, mode)
    seen.add(key)
    actual_fail = int(row["fail"])
    actual_error = int(row["error"])
    exp = expected.get(key, {"expected_fail": 0, "expected_error": 0, "notes": ""})
    expected_fail = exp["expected_fail"]
    expected_error = exp["expected_error"]
    unexpected_fail = max(actual_fail - expected_fail, 0)
    unexpected_error = max(actual_error - expected_error, 0)
    totals["actual_fail"] += actual_fail
    totals["actual_error"] += actual_error
    totals["expected_fail"] += expected_fail
    totals["expected_error"] += expected_error
    totals["unexpected_fail"] += unexpected_fail
    totals["unexpected_error"] += unexpected_error
    rows.append(
        {
            "suite": suite,
            "mode": mode,
            "fail": actual_fail,
            "error": actual_error,
            "expected_fail": expected_fail,
            "expected_error": expected_error,
            "unexpected_fail": unexpected_fail,
            "unexpected_error": unexpected_error,
            "within_budget": "yes" if unexpected_fail == 0 and unexpected_error == 0 else "no",
            "notes": exp["notes"],
        }
    )

unused_expectations = []
for suite, mode in sorted(expected.keys() - seen):
  unused_expectations.append({"suite": suite, "mode": mode})

with expected_summary_path.open("w", newline="") as f:
  writer = csv.DictWriter(
      f,
      fieldnames=[
          "suite",
          "mode",
          "fail",
          "error",
          "expected_fail",
          "expected_error",
          "unexpected_fail",
          "unexpected_error",
          "within_budget",
          "notes",
      ],
      delimiter='\t',
  )
  writer.writeheader()
  for row in rows:
    writer.writerow(row)

try:
  payload = json.loads(json_summary_path.read_text())
except Exception:
  payload = {}

payload["expected_failures"] = {
    "file": str(expected_path) if expected_path is not None else "",
    "rows": rows,
    "totals": totals,
    "unused_expectations": unused_expectations,
}
json_summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

print(f"expected-failures summary: {expected_summary_path}")
print(
    "expected-failures totals: "
    f"actual_fail={totals['actual_fail']} actual_error={totals['actual_error']} "
    f"expected_fail={totals['expected_fail']} expected_error={totals['expected_error']} "
    f"unexpected_fail={totals['unexpected_fail']} unexpected_error={totals['unexpected_error']}"
)
if unused_expectations:
  print("expected-failures unused entries:")
  for item in unused_expectations:
    print(f"  {item['suite']} {item['mode']}")

if prune_path is not None:
  if expected_path is None:
    raise SystemExit("--prune-expected-failures-file requires expected failures to be loaded")
  pruned_rows = []
  dropped_rows = []
  dropped_unused = 0
  for (suite, mode), exp in expected.items():
    if prune_drop_unused and (suite, mode) not in seen:
      dropped_unused += 1
      dropped_rows.append(
          {
              "suite": suite,
              "mode": mode,
              "expected_fail": exp["expected_fail"],
              "expected_error": exp["expected_error"],
              "notes": exp.get("notes", ""),
              "drop_reason": "unused",
          }
      )
      continue
    pruned_rows.append(
        {
            "suite": suite,
            "mode": mode,
            "expected_fail": exp["expected_fail"],
            "expected_error": exp["expected_error"],
            "notes": exp.get("notes", ""),
        }
    )
  prune_path.parent.mkdir(parents=True, exist_ok=True)
  if expectations_dry_run:
    print(f"dry-run: would prune expected-failures file: {prune_path}")
    emit_dry_run_report(
        {
            "operation": "prune_expected_failures",
            "target_file": str(prune_path),
            "kept_rows": len(pruned_rows),
            "dropped_unused": dropped_unused,
            "kept_rows_sample": sample_rows(pruned_rows),
            "dropped_rows_sample": sample_rows(dropped_rows),
        }
    )
  else:
    with prune_path.open("w", newline="") as f:
      writer = csv.DictWriter(
          f,
          fieldnames=["suite", "mode", "expected_fail", "expected_error", "notes"],
          delimiter="\t",
      )
      writer.writeheader()
      for row in pruned_rows:
        writer.writerow(row)
    print(f"pruned expected-failures file: {prune_path}")
  print(
      "pruned expected-failures rows: "
      f"kept={len(pruned_rows)} dropped_unused={dropped_unused}"
  )

if fail_on_unexpected and (
    totals["unexpected_fail"] > 0 or totals["unexpected_error"] > 0
):
  print("unexpected failure budget overruns:")
  for row in rows:
    if row["unexpected_fail"] > 0 or row["unexpected_error"] > 0:
      print(
          f"  {row['suite']} {row['mode']}: "
          f"fail {row['fail']} (expected {row['expected_fail']}), "
          f"error {row['error']} (expected {row['expected_error']})"
      )
  raise SystemExit(1)

if fail_on_unused and unused_expectations:
  print("unused expected-failures entries:")
  for item in unused_expectations:
    print(f"  {item['suite']} {item['mode']}")
  raise SystemExit(1)
PY
fi

if [[ -n "$REFRESH_EXPECTED_FAILURES_FILE" ]]; then
  OUT_DIR="$OUT_DIR" \
  SUMMARY_FILE="$OUT_DIR/summary.tsv" \
  EXPECTATIONS_DRY_RUN="$EXPECTATIONS_DRY_RUN" \
  EXPECTATIONS_DRY_RUN_RUN_ID="$RUN_ID" \
  EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS="$EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS" \
  EXPECTATIONS_DRY_RUN_REPORT_JSONL="$EXPECTATIONS_DRY_RUN_REPORT_JSONL" \
  REFRESH_EXPECTED_FAILURES_FILE="$REFRESH_EXPECTED_FAILURES_FILE" \
  REFRESH_EXPECTED_FAILURES_INCLUDE_SUITE_REGEX="$REFRESH_EXPECTED_FAILURES_INCLUDE_SUITE_REGEX" \
  REFRESH_EXPECTED_FAILURES_INCLUDE_MODE_REGEX="$REFRESH_EXPECTED_FAILURES_INCLUDE_MODE_REGEX" \
  python3 - <<'PY'
import csv
import json
import os
import re
from pathlib import Path

summary_path = Path(os.environ["SUMMARY_FILE"])
out_path = Path(os.environ["REFRESH_EXPECTED_FAILURES_FILE"])
expectations_dry_run = os.environ.get("EXPECTATIONS_DRY_RUN", "0") == "1"
dry_run_run_id = os.environ.get("EXPECTATIONS_DRY_RUN_RUN_ID", "")
max_sample_rows = int(
    os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS", "5")
)
dry_run_report_jsonl_raw = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_JSONL", "")
dry_run_report_jsonl_path = Path(dry_run_report_jsonl_raw) if dry_run_report_jsonl_raw else None
suite_filter_raw = os.environ.get("REFRESH_EXPECTED_FAILURES_INCLUDE_SUITE_REGEX", "")
mode_filter_raw = os.environ.get("REFRESH_EXPECTED_FAILURES_INCLUDE_MODE_REGEX", "")

def emit_dry_run_report(payload):
  if dry_run_report_jsonl_path is None:
    return
  payload = dict(payload)
  payload.setdefault("run_id", dry_run_run_id)
  dry_run_report_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
  with dry_run_report_jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")

def sample_rows(rows):
  if max_sample_rows <= 0:
    return []
  return rows[:max_sample_rows]

def compile_optional_regex(raw: str, field: str):
  if not raw:
    return None
  try:
    return re.compile(raw)
  except re.error as ex:
    raise SystemExit(f"invalid {field}: {ex}")

suite_filter = compile_optional_regex(
    suite_filter_raw, "--refresh-expected-failures-include-suite-regex"
)
mode_filter = compile_optional_regex(
    mode_filter_raw, "--refresh-expected-failures-include-mode-regex"
)

existing_notes = {}
if out_path.exists():
  with out_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
      suite = (row.get("suite") or "").strip()
      mode = (row.get("mode") or "").strip()
      if not suite or not mode:
        continue
      existing_notes[(suite, mode)] = row.get("notes", "")

rows = []
with summary_path.open() as f:
  reader = csv.DictReader(f, delimiter="\t")
  for row in reader:
    suite = row["suite"]
    mode = row["mode"]
    if suite_filter is not None and suite_filter.search(suite) is None:
      continue
    if mode_filter is not None and mode_filter.search(mode) is None:
      continue
    rows.append(
        {
            "suite": suite,
            "mode": mode,
            "expected_fail": int(row.get("fail", "0") or 0),
            "expected_error": int(row.get("error", "0") or 0),
            "notes": existing_notes.get((suite, mode), ""),
        }
    )

rows.sort(key=lambda r: (r["suite"], r["mode"]))
out_path.parent.mkdir(parents=True, exist_ok=True)
if expectations_dry_run:
  print(f"dry-run: would refresh expected-failures file: {out_path}")
  emit_dry_run_report(
      {
          "operation": "refresh_expected_failures",
          "target_file": str(out_path),
          "output_rows": len(rows),
          "output_rows_sample": sample_rows(rows),
          "suite_filter": suite_filter_raw,
          "mode_filter": mode_filter_raw,
      }
  )
else:
  with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["suite", "mode", "expected_fail", "expected_error", "notes"],
        delimiter="\t",
    )
    writer.writeheader()
    for row in rows:
      writer.writerow(row)
  print(f"refreshed expected-failures file: {out_path}")
print(f"refreshed expected-failures rows: {len(rows)}")
PY
fi

if [[ -n "$EXPECTED_FAILURE_CASES_FILE" || \
      "$FAIL_ON_UNEXPECTED_FAILURE_CASES" == "1" || \
      "$FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES" == "1" || \
      "$FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES" == "1" || \
      -n "$PRUNE_EXPECTED_FAILURE_CASES_FILE" ]]; then
  OUT_DIR="$OUT_DIR" \
  JSON_SUMMARY_FILE="$JSON_SUMMARY_FILE" \
  EXPECTED_FAILURE_CASES_FILE="$EXPECTED_FAILURE_CASES_FILE" \
  FAIL_ON_UNEXPECTED_FAILURE_CASES="$FAIL_ON_UNEXPECTED_FAILURE_CASES" \
  FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES="$FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES" \
  FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES="$FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES" \
  EXPECTATIONS_DRY_RUN="$EXPECTATIONS_DRY_RUN" \
  EXPECTATIONS_DRY_RUN_RUN_ID="$RUN_ID" \
  EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS="$EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS" \
  EXPECTATIONS_DRY_RUN_REPORT_JSONL="$EXPECTATIONS_DRY_RUN_REPORT_JSONL" \
  PRUNE_EXPECTED_FAILURE_CASES_FILE="$PRUNE_EXPECTED_FAILURE_CASES_FILE" \
  PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED="$PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED" \
  PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED="$PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED" \
  python3 - <<'PY'
import csv
import datetime as dt
import json
import os
import re
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
json_summary_path = Path(os.environ["JSON_SUMMARY_FILE"])
expected_file_raw = os.environ.get("EXPECTED_FAILURE_CASES_FILE", "")
expected_path = Path(expected_file_raw) if expected_file_raw else None
fail_on_unexpected = os.environ.get("FAIL_ON_UNEXPECTED_FAILURE_CASES", "0") == "1"
fail_on_expired = os.environ.get("FAIL_ON_EXPIRED_EXPECTED_FAILURE_CASES", "0") == "1"
fail_on_unmatched = os.environ.get("FAIL_ON_UNMATCHED_EXPECTED_FAILURE_CASES", "0") == "1"
expectations_dry_run = os.environ.get("EXPECTATIONS_DRY_RUN", "0") == "1"
dry_run_run_id = os.environ.get("EXPECTATIONS_DRY_RUN_RUN_ID", "")
max_sample_rows = int(
    os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS", "5")
)
dry_run_report_jsonl_raw = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_JSONL", "")
prune_file_raw = os.environ.get("PRUNE_EXPECTED_FAILURE_CASES_FILE", "")
prune_path = Path(prune_file_raw) if prune_file_raw else None
dry_run_report_jsonl_path = Path(dry_run_report_jsonl_raw) if dry_run_report_jsonl_raw else None
prune_drop_unmatched = (
    os.environ.get("PRUNE_EXPECTED_FAILURE_CASES_DROP_UNMATCHED", "0") == "1"
)
prune_drop_expired = (
    os.environ.get("PRUNE_EXPECTED_FAILURE_CASES_DROP_EXPIRED", "0") == "1"
)
today = dt.date.today()

case_summary_path = out_dir / "expected-failure-cases-summary.tsv"
unexpected_path = out_dir / "unexpected-failure-cases.tsv"

def emit_dry_run_report(payload):
  if dry_run_report_jsonl_path is None:
    return
  payload = dict(payload)
  payload.setdefault("run_id", dry_run_run_id)
  dry_run_report_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
  with dry_run_report_jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")

def sample_rows(rows):
  if max_sample_rows <= 0:
    return []
  return rows[:max_sample_rows]

fail_like_statuses = {"FAIL", "ERROR", "XFAIL", "XPASS", "EFAIL", "TIMEOUT", "UNKNOWN"}


def extract_diag_tag(path: str) -> str:
  if "#" not in path:
    return ""
  candidate = path.rsplit("#", 1)[1].strip()
  if re.fullmatch(r"[A-Z0-9_]+", candidate):
    return candidate
  return ""


def derive_base_diag(base: str, path: str) -> str:
  diag = extract_diag_tag(path)
  if base and diag:
    return f"{base}#{diag}"
  return base


result_sources = [
    ("sv-tests", "BMC", out_dir / "sv-tests-bmc-results.txt"),
    ("sv-tests", "LEC", out_dir / "sv-tests-lec-results.txt"),
    ("verilator-verification", "BMC", out_dir / "verilator-bmc-results.txt"),
    ("verilator-verification", "LEC", out_dir / "verilator-lec-results.txt"),
    ("yosys/tests/sva", "BMC", out_dir / "yosys-bmc-results.txt"),
    ("yosys/tests/sva", "LEC", out_dir / "yosys-lec-results.txt"),
    ("opentitan", "LEC", out_dir / "opentitan-lec-results.txt"),
    ("opentitan", "LEC_STRICT", out_dir / "opentitan-lec-strict-results.txt"),
    ("opentitan", "E2E", out_dir / "opentitan-e2e-results.txt"),
    ("opentitan", "E2E_STRICT", out_dir / "opentitan-e2e-strict-results.txt"),
    ("opentitan", "E2E_MODE_DIFF", out_dir / "opentitan-e2e-mode-diff-results.txt"),
    ("", "", out_dir / "avip-results.txt"),
]
detailed_source_pairs = {
    (suite, mode) for suite, mode, _ in result_sources if suite and mode
}
detailed_pairs_observed = set()

observed = []
for default_suite, default_mode, path in result_sources:
  if not path.exists():
    continue
  with path.open() as f:
    for line in f:
      line = line.rstrip("\n")
      if not line:
        continue
      parts = line.split("\t")
      status = parts[0].strip().upper() if parts else ""
      if status not in fail_like_statuses:
        continue
      suite = (
          parts[3].strip() if len(parts) > 3 and parts[3].strip() else default_suite
      )
      mode = (
          parts[4].strip() if len(parts) > 4 and parts[4].strip() else default_mode
      )
      if not suite or not mode:
        continue
      base = parts[1].strip() if len(parts) > 1 else ""
      file_path = parts[2].strip() if len(parts) > 2 else ""
      diag = extract_diag_tag(file_path)
      detailed_pairs_observed.add((suite, mode))
      observed.append(
          {
              "suite": suite,
              "mode": mode,
              "status": status,
              "base": base,
              "base_diag": derive_base_diag(base, file_path),
              "diag": diag,
              "path": file_path,
          }
      )

summary_path = out_dir / "summary.tsv"
if summary_path.exists():
  with summary_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
      suite = row.get("suite", "")
      mode = row.get("mode", "")
      if not suite or not mode:
        continue
      if (suite, mode) in detailed_source_pairs or (suite, mode) in detailed_pairs_observed:
        continue
      summary = row.get("summary", "")
      try:
        fail_count = int(row.get("fail", "0"))
      except Exception:
        fail_count = 0
      try:
        error_count = int(row.get("error", "0"))
      except Exception:
        error_count = 0
      try:
        xfail_count = int(row.get("xfail", "0"))
      except Exception:
        xfail_count = 0
      try:
        xpass_count = int(row.get("xpass", "0"))
      except Exception:
        xpass_count = 0
      if fail_count > 0:
        observed.append(
            {
                "suite": suite,
                "mode": mode,
                "status": "FAIL",
                "base": "__aggregate__",
                "base_diag": "__aggregate__",
                "diag": "",
                "path": summary,
            }
        )
      if error_count > 0:
        observed.append(
            {
                "suite": suite,
                "mode": mode,
                "status": "ERROR",
                "base": "__aggregate__",
                "base_diag": "__aggregate__",
                "diag": "",
                "path": summary,
            }
        )
      if xfail_count > 0:
        observed.append(
            {
                "suite": suite,
                "mode": mode,
                "status": "XFAIL",
                "base": "__aggregate__",
                "base_diag": "__aggregate__",
                "diag": "",
                "path": summary,
            }
        )
      if xpass_count > 0:
        observed.append(
            {
                "suite": suite,
                "mode": mode,
                "status": "XPASS",
                "base": "__aggregate__",
                "base_diag": "__aggregate__",
                "diag": "",
                "path": summary,
            }
        )

expected_rows = []
if expected_path is not None:
  with expected_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    required_cols = {"suite", "mode", "id"}
    fieldnames = set(reader.fieldnames or [])
    if not required_cols.issubset(fieldnames):
      missing = sorted(required_cols - fieldnames)
      raise SystemExit(
          "invalid expected-failure-cases file: "
          f"missing required columns: {', '.join(missing)}"
      )
    seen = set()
    for idx, row in enumerate(reader):
      suite = row.get("suite", "").strip()
      mode = row.get("mode", "").strip()
      case_id = row.get("id", "").strip()
      if not suite or not mode or not case_id:
        raise SystemExit(
            "invalid expected-failure-cases row: "
            f"suite/mode/id must be non-empty at data row {idx + 1}"
        )
      id_kind = row.get("id_kind", "base").strip().lower() or "base"
      if id_kind not in {"base", "base_diag", "path", "aggregate", "base_regex", "base_diag_regex", "path_regex"}:
        raise SystemExit(
            "invalid expected-failure-cases row for "
            f"suite={suite} mode={mode} id={case_id}: "
            "id_kind must be one of "
            "base,base_diag,path,aggregate,base_regex,base_diag_regex,path_regex"
        )
      id_re = None
      if id_kind in {"base_regex", "base_diag_regex", "path_regex"}:
        try:
          id_re = re.compile(case_id)
        except re.error as ex:
          raise SystemExit(
              "invalid expected-failure-cases row for "
              f"suite={suite} mode={mode} id={case_id}: "
              f"invalid regex ({ex})"
          )
      status = row.get("status", "ANY").strip().upper() or "ANY"
      if status != "ANY" and status not in fail_like_statuses:
        raise SystemExit(
            "invalid expected-failure-cases row for "
            f"suite={suite} mode={mode} id={case_id}: "
            f"unsupported status '{status}'"
        )
      expires_on = row.get("expires_on", "").strip()
      expires_date = None
      if expires_on:
        try:
          expires_date = dt.date.fromisoformat(expires_on)
        except Exception:
          raise SystemExit(
              "invalid expected-failure-cases row for "
              f"suite={suite} mode={mode} id={case_id}: "
              f"invalid expires_on '{expires_on}' (expected YYYY-MM-DD)"
          )
      key = (suite, mode, id_kind, case_id, status)
      if key in seen:
        raise SystemExit(
            "invalid expected-failure-cases file: "
            f"duplicate row for suite={suite} mode={mode} "
            f"id_kind={id_kind} id={case_id} status={status}"
        )
      seen.add(key)
      expected_rows.append(
          {
              "suite": suite,
              "mode": mode,
              "id_kind": id_kind,
              "id": case_id,
              "id_re": id_re,
              "status": status,
              "expires_on": expires_on,
              "expires_date": expires_date,
              "reason": row.get("reason", ""),
          }
      )

matched_observed_idx = set()
expected_summary_rows = []
for row in expected_rows:
  matches = []
  for idx, obs in enumerate(observed):
    if obs["suite"] != row["suite"] or obs["mode"] != row["mode"]:
      continue
    if row["id_kind"] == "base":
      if obs["base"] != row["id"]:
        continue
    elif row["id_kind"] == "base_diag":
      if obs["base_diag"] != row["id"]:
        continue
    elif row["id_kind"] == "base_diag_regex":
      if row["id_re"] is None or row["id_re"].search(obs["base_diag"]) is None:
        continue
    elif row["id_kind"] == "path":
      if obs["path"] != row["id"]:
        continue
    elif row["id_kind"] == "base_regex":
      if row["id_re"] is None or row["id_re"].search(obs["base"]) is None:
        continue
    elif row["id_kind"] == "path_regex":
      if row["id_re"] is None or row["id_re"].search(obs["path"]) is None:
        continue
    else:
      if row["id"] != "__aggregate__":
        continue
    if row["status"] != "ANY" and obs["status"] != row["status"]:
      continue
    matches.append(idx)
  for idx in matches:
    matched_observed_idx.add(idx)
  expired = "yes" if row["expires_date"] is not None and row["expires_date"] < today else "no"
  expected_summary_rows.append(
      {
          "suite": row["suite"],
          "mode": row["mode"],
          "id_kind": row["id_kind"],
          "id": row["id"],
          "status": row["status"],
          "expires_on": row["expires_on"],
          "matched_count": len(matches),
          "expired": expired,
          "reason": row["reason"],
      }
  )

unexpected_observed = []
for idx, obs in enumerate(observed):
  if idx in matched_observed_idx:
    continue
  unexpected_observed.append(obs)

with case_summary_path.open("w", newline="") as f:
  writer = csv.DictWriter(
      f,
      fieldnames=[
          "suite",
          "mode",
          "id_kind",
          "id",
          "status",
          "expires_on",
          "matched_count",
          "expired",
          "reason",
      ],
      delimiter="\t",
  )
  writer.writeheader()
  for row in expected_summary_rows:
    writer.writerow(row)

with unexpected_path.open("w", newline="") as f:
  writer = csv.DictWriter(
      f,
      fieldnames=["suite", "mode", "status", "base", "path"],
      delimiter="\t",
  )
  writer.writeheader()
  for row in unexpected_observed:
    writer.writerow(
        {
            "suite": row.get("suite", ""),
            "mode": row.get("mode", ""),
            "status": row.get("status", ""),
            "base": row.get("base", ""),
            "path": row.get("path", ""),
        }
    )

expired_rows = [row for row in expected_summary_rows if row["expired"] == "yes"]
unmatched_rows = [row for row in expected_summary_rows if row["matched_count"] == 0]
totals = {
    "observed_fail_like": len(observed),
    "matched_expected": len(matched_observed_idx),
    "unexpected_observed": len(unexpected_observed),
    "expected_rows": len(expected_summary_rows),
    "unmatched_expected": len(unmatched_rows),
    "expired_expected": len(expired_rows),
}

try:
  payload = json.loads(json_summary_path.read_text())
except Exception:
  payload = {}

payload["expected_failure_cases"] = {
    "file": str(expected_path) if expected_path is not None else "",
    "rows": expected_summary_rows,
    "unexpected_observed": unexpected_observed,
    "totals": totals,
}
json_summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

print(f"expected-failure-cases summary: {case_summary_path}")
print(f"unexpected-failure-cases summary: {unexpected_path}")
print(
    "expected-failure-cases totals: "
    f"observed_fail_like={totals['observed_fail_like']} "
    f"matched_expected={totals['matched_expected']} "
    f"unexpected_observed={totals['unexpected_observed']} "
    f"expired_expected={totals['expired_expected']} "
    f"unmatched_expected={totals['unmatched_expected']}"
)

if prune_path is not None:
  if expected_path is None:
    raise SystemExit(
        "--prune-expected-failure-cases-file requires expected cases to be loaded"
    )
  pruned_rows = []
  dropped_rows = []
  dropped_unmatched = 0
  dropped_expired = 0
  for row, summary_row in zip(expected_rows, expected_summary_rows):
    is_unmatched = summary_row["matched_count"] == 0
    is_expired = summary_row["expired"] == "yes"
    drop = False
    drop_reasons = []
    if prune_drop_unmatched and is_unmatched:
      drop = True
      dropped_unmatched += 1
      drop_reasons.append("unmatched")
    if prune_drop_expired and is_expired:
      drop = True
      dropped_expired += 1
      drop_reasons.append("expired")
    if drop:
      dropped_rows.append(
          {
              "suite": row["suite"],
              "mode": row["mode"],
              "id": row["id"],
              "id_kind": row["id_kind"],
              "status": row["status"],
              "expires_on": row["expires_on"],
              "reason": row["reason"],
              "drop_reasons": ",".join(drop_reasons),
          }
      )
      continue
    pruned_rows.append(
        {
            "suite": row["suite"],
            "mode": row["mode"],
            "id": row["id"],
            "id_kind": row["id_kind"],
            "status": row["status"],
            "expires_on": row["expires_on"],
            "reason": row["reason"],
        }
    )

  prune_path.parent.mkdir(parents=True, exist_ok=True)
  if expectations_dry_run:
    print(f"dry-run: would prune expected-failure-cases file: {prune_path}")
    emit_dry_run_report(
        {
            "operation": "prune_expected_failure_cases",
            "target_file": str(prune_path),
            "kept_rows": len(pruned_rows),
            "dropped_unmatched": dropped_unmatched,
            "dropped_expired": dropped_expired,
            "kept_rows_sample": sample_rows(pruned_rows),
            "dropped_rows_sample": sample_rows(dropped_rows),
        }
    )
  else:
    with prune_path.open("w", newline="") as f:
      writer = csv.DictWriter(
          f,
          fieldnames=["suite", "mode", "id", "id_kind", "status", "expires_on", "reason"],
          delimiter="\t",
      )
      writer.writeheader()
      for row in pruned_rows:
        writer.writerow(row)
    print(f"pruned expected-failure-cases file: {prune_path}")
  print(
      "pruned expected-failure-cases rows: "
      f"kept={len(pruned_rows)} dropped_unmatched={dropped_unmatched} "
      f"dropped_expired={dropped_expired}"
  )

if fail_on_unexpected and unexpected_observed:
  print("unexpected observed failure cases:")
  for row in unexpected_observed:
    print(
        f"  {row['suite']} {row['mode']} {row['status']} "
        f"base={row['base']} path={row['path']}"
    )
  raise SystemExit(1)

if fail_on_expired and expired_rows:
  print("expired expected failure cases:")
  for row in expired_rows:
    print(
        f"  {row['suite']} {row['mode']} id_kind={row['id_kind']} "
        f"id={row['id']} expires_on={row['expires_on']} matched_count={row['matched_count']}"
      )
  raise SystemExit(1)

if fail_on_unmatched and unmatched_rows:
  print("unmatched expected failure cases:")
  for row in unmatched_rows:
    print(
        f"  {row['suite']} {row['mode']} id_kind={row['id_kind']} "
        f"id={row['id']} status={row['status']}"
    )
  raise SystemExit(1)
PY
fi

if [[ -n "$REFRESH_EXPECTED_FAILURE_CASES_FILE" ]]; then
  OUT_DIR="$OUT_DIR" \
  EXPECTATIONS_DRY_RUN="$EXPECTATIONS_DRY_RUN" \
  EXPECTATIONS_DRY_RUN_RUN_ID="$RUN_ID" \
  EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS="$EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS" \
  EXPECTATIONS_DRY_RUN_REPORT_JSONL="$EXPECTATIONS_DRY_RUN_REPORT_JSONL" \
  REFRESH_EXPECTED_FAILURE_CASES_FILE="$REFRESH_EXPECTED_FAILURE_CASES_FILE" \
  REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON="$REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON" \
  REFRESH_EXPECTED_FAILURE_CASES_COLLAPSE_STATUS_ANY="$REFRESH_EXPECTED_FAILURE_CASES_COLLAPSE_STATUS_ANY" \
  REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_SUITE_REGEX="$REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_SUITE_REGEX" \
  REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_MODE_REGEX="$REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_MODE_REGEX" \
  REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_STATUS_REGEX="$REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_STATUS_REGEX" \
  REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_ID_REGEX="$REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_ID_REGEX" \
  python3 - <<'PY'
import csv
import datetime as dt
import json
import os
import re
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
out_path = Path(os.environ["REFRESH_EXPECTED_FAILURE_CASES_FILE"])
expectations_dry_run = os.environ.get("EXPECTATIONS_DRY_RUN", "0") == "1"
dry_run_run_id = os.environ.get("EXPECTATIONS_DRY_RUN_RUN_ID", "")
max_sample_rows = int(
    os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_MAX_SAMPLE_ROWS", "5")
)
dry_run_report_jsonl_raw = os.environ.get("EXPECTATIONS_DRY_RUN_REPORT_JSONL", "")
dry_run_report_jsonl_path = Path(dry_run_report_jsonl_raw) if dry_run_report_jsonl_raw else None
default_expires = os.environ.get("REFRESH_EXPECTED_FAILURE_CASES_DEFAULT_EXPIRES_ON", "").strip()
collapse_status_any = (
    os.environ.get("REFRESH_EXPECTED_FAILURE_CASES_COLLAPSE_STATUS_ANY", "0") == "1"
)
suite_filter_raw = os.environ.get(
    "REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_SUITE_REGEX", ""
)
mode_filter_raw = os.environ.get(
    "REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_MODE_REGEX", ""
)
status_filter_raw = os.environ.get(
    "REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_STATUS_REGEX", ""
)
id_filter_raw = os.environ.get("REFRESH_EXPECTED_FAILURE_CASES_INCLUDE_ID_REGEX", "")

def emit_dry_run_report(payload):
  if dry_run_report_jsonl_path is None:
    return
  payload = dict(payload)
  payload.setdefault("run_id", dry_run_run_id)
  dry_run_report_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
  with dry_run_report_jsonl_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, sort_keys=True) + "\n")

def sample_rows(rows):
  if max_sample_rows <= 0:
    return []
  return rows[:max_sample_rows]

def compile_optional_regex(raw: str, field: str):
  if not raw:
    return None
  try:
    return re.compile(raw)
  except re.error as ex:
    raise SystemExit(f"invalid {field}: {ex}")

suite_filter = compile_optional_regex(
    suite_filter_raw, "--refresh-expected-failure-cases-include-suite-regex"
)
mode_filter = compile_optional_regex(
    mode_filter_raw, "--refresh-expected-failure-cases-include-mode-regex"
)
status_filter = compile_optional_regex(
    status_filter_raw, "--refresh-expected-failure-cases-include-status-regex"
)
id_filter = compile_optional_regex(
    id_filter_raw, "--refresh-expected-failure-cases-include-id-regex"
)

if default_expires:
  try:
    dt.date.fromisoformat(default_expires)
  except Exception:
    raise SystemExit(
        "invalid --refresh-expected-failure-cases-default-expires-on: "
        f"{default_expires}"
    )

existing_meta = {}
if out_path.exists():
  with out_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
      suite = (row.get("suite") or "").strip()
      mode = (row.get("mode") or "").strip()
      case_id = (row.get("id") or "").strip()
      id_kind = (row.get("id_kind") or "base").strip().lower() or "base"
      status = (row.get("status") or "ANY").strip().upper() or "ANY"
      if not suite or not mode or not case_id:
        continue
      key = (suite, mode, id_kind, case_id, status)
      existing_meta[key] = {
          "expires_on": (row.get("expires_on") or "").strip(),
          "reason": row.get("reason", ""),
      }

fail_like_statuses = {"FAIL", "ERROR", "XFAIL", "XPASS", "EFAIL", "TIMEOUT", "UNKNOWN"}


def extract_diag_tag(path: str) -> str:
  if "#" not in path:
    return ""
  candidate = path.rsplit("#", 1)[1].strip()
  if re.fullmatch(r"[A-Z0-9_]+", candidate):
    return candidate
  return ""


def derive_base_diag(base: str, path: str) -> str:
  diag = extract_diag_tag(path)
  if base and diag:
    return f"{base}#{diag}"
  return base


result_sources = [
    ("sv-tests", "BMC", out_dir / "sv-tests-bmc-results.txt"),
    ("sv-tests", "LEC", out_dir / "sv-tests-lec-results.txt"),
    ("verilator-verification", "BMC", out_dir / "verilator-bmc-results.txt"),
    ("verilator-verification", "LEC", out_dir / "verilator-lec-results.txt"),
    ("yosys/tests/sva", "BMC", out_dir / "yosys-bmc-results.txt"),
    ("yosys/tests/sva", "LEC", out_dir / "yosys-lec-results.txt"),
    ("opentitan", "LEC", out_dir / "opentitan-lec-results.txt"),
    ("opentitan", "LEC_STRICT", out_dir / "opentitan-lec-strict-results.txt"),
    ("opentitan", "E2E", out_dir / "opentitan-e2e-results.txt"),
    ("opentitan", "E2E_STRICT", out_dir / "opentitan-e2e-strict-results.txt"),
    ("opentitan", "E2E_MODE_DIFF", out_dir / "opentitan-e2e-mode-diff-results.txt"),
    ("", "", out_dir / "avip-results.txt"),
]
detailed_pairs_observed = set()

observed = []
for default_suite, default_mode, path in result_sources:
  if not path.exists():
    continue
  with path.open() as f:
    for line in f:
      line = line.rstrip("\n")
      if not line:
        continue
      parts = line.split("\t")
      status = parts[0].strip().upper() if parts else ""
      if status not in fail_like_statuses:
        continue
      suite = (
          parts[3].strip() if len(parts) > 3 and parts[3].strip() else default_suite
      )
      mode = (
          parts[4].strip() if len(parts) > 4 and parts[4].strip() else default_mode
      )
      if not suite or not mode:
        continue
      base = parts[1].strip() if len(parts) > 1 else ""
      file_path = parts[2].strip() if len(parts) > 2 else ""
      diag = extract_diag_tag(file_path)
      detailed_pairs_observed.add((suite, mode))
      observed.append(
          {
              "suite": suite,
              "mode": mode,
              "status": status,
              "base": base,
              "base_diag": derive_base_diag(base, file_path),
              "diag": diag,
              "path": file_path,
          }
      )

summary_path = out_dir / "summary.tsv"
if summary_path.exists():
  with summary_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
      suite = row.get("suite", "")
      mode = row.get("mode", "")
      if not suite or not mode:
        continue
      if (suite, mode) in detailed_pairs_observed:
        continue
      summary = row.get("summary", "")
      for summary_key, status in (
          ("fail", "FAIL"),
          ("error", "ERROR"),
          ("xfail", "XFAIL"),
          ("xpass", "XPASS"),
      ):
        try:
          count = int(row.get(summary_key, "0"))
        except Exception:
          count = 0
        if count <= 0:
          continue
        observed.append(
            {
                "suite": suite,
                "mode": mode,
                "status": status,
                "base": "__aggregate__",
                "base_diag": "__aggregate__",
                "diag": "",
                "path": summary,
            }
        )

refreshed = []
seen = set()

def derive_case_id(obs_row):
  if obs_row["base"] == "__aggregate__":
    return ("aggregate", "__aggregate__")
  if obs_row.get("diag") and obs_row["base"]:
    return ("base_diag", obs_row["base_diag"])
  if obs_row["base"]:
    return ("base", obs_row["base"])
  return ("path", obs_row["path"])

def include_obs(obs_row, id_kind: str, case_id: str):
  if suite_filter is not None and suite_filter.search(obs_row["suite"]) is None:
    return False
  if mode_filter is not None and mode_filter.search(obs_row["mode"]) is None:
    return False
  if status_filter is not None and status_filter.search(obs_row["status"]) is None:
    return False
  if id_filter is not None and id_filter.search(case_id) is None:
    return False
  return True

if collapse_status_any:
  grouped = {}
  for obs in observed:
    id_kind, case_id = derive_case_id(obs)
    if not include_obs(obs, id_kind, case_id):
      continue
    base_key = (obs["suite"], obs["mode"], id_kind, case_id)
    grouped.setdefault(base_key, set()).add(obs["status"])
  for base_key, statuses in grouped.items():
    suite, mode, id_kind, case_id = base_key
    key = (suite, mode, id_kind, case_id, "ANY")
    if key in seen:
      continue
    seen.add(key)
    meta = existing_meta.get(key, {})
    if not meta:
      for status in sorted(statuses):
        exact_key = (suite, mode, id_kind, case_id, status)
        if exact_key in existing_meta:
          meta = existing_meta[exact_key]
          break
    refreshed.append(
        {
            "suite": suite,
            "mode": mode,
            "id": case_id,
            "id_kind": id_kind,
            "status": "ANY",
            "expires_on": meta.get("expires_on", "") or default_expires,
            "reason": meta.get("reason", ""),
        }
    )
else:
  for obs in observed:
    id_kind, case_id = derive_case_id(obs)
    if not include_obs(obs, id_kind, case_id):
      continue
    key = (obs["suite"], obs["mode"], id_kind, case_id, obs["status"])
    if key in seen:
      continue
    seen.add(key)
    meta = existing_meta.get(key, {})
    refreshed.append(
        {
            "suite": obs["suite"],
            "mode": obs["mode"],
            "id": case_id,
            "id_kind": id_kind,
            "status": obs["status"],
            "expires_on": meta.get("expires_on", "") or default_expires,
            "reason": meta.get("reason", ""),
        }
    )

refreshed.sort(key=lambda r: (r["suite"], r["mode"], r["id_kind"], r["id"], r["status"]))
out_path.parent.mkdir(parents=True, exist_ok=True)
if expectations_dry_run:
  print(f"dry-run: would refresh expected-failure-cases file: {out_path}")
  emit_dry_run_report(
      {
          "operation": "refresh_expected_failure_cases",
          "target_file": str(out_path),
          "output_rows": len(refreshed),
          "output_rows_sample": sample_rows(refreshed),
          "collapse_status_any": collapse_status_any,
          "suite_filter": suite_filter_raw,
          "mode_filter": mode_filter_raw,
          "status_filter": status_filter_raw,
          "id_filter": id_filter_raw,
      }
  )
else:
  with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["suite", "mode", "id", "id_kind", "status", "expires_on", "reason"],
        delimiter="\t",
    )
    writer.writeheader()
    for row in refreshed:
      writer.writerow(row)
  print(f"refreshed expected-failure-cases file: {out_path}")
print(f"refreshed expected-failure-cases rows: {len(refreshed)}")
PY
fi

if [[ "$UPDATE_BASELINES" == "1" ]]; then
  OUT_DIR="$OUT_DIR" DATE_STR="$DATE_STR" BASELINE_FILE="$BASELINE_FILE" PLAN_FILE="$PLAN_FILE" python3 - <<'PY'
import csv
import os
import re
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
summary_path = out_dir / "summary.tsv"
baseline_path = Path(os.environ["BASELINE_FILE"])
plan_path = Path(os.environ["PLAN_FILE"])

date_str = os.environ["DATE_STR"]

rows = []
with summary_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        rows.append(row)

def parse_result_summary(summary: str):
    parsed = {}
    for match in re.finditer(r"([a-z][a-z0-9_]*)=([0-9]+)", summary):
        parsed[match.group(1)] = int(match.group(2))
    return parsed

fail_like_statuses = {"FAIL", "ERROR", "XFAIL", "XPASS", "EFAIL", "TIMEOUT", "UNKNOWN"}

def extract_diag_tag(path: str) -> str:
    if "#" not in path:
        return ""
    candidate = path.rsplit("#", 1)[1].strip()
    if re.fullmatch(r"[A-Z0-9_]+", candidate):
        return candidate
    return ""

def compose_case_id(base: str, path: str) -> str:
    diag = extract_diag_tag(path)
    if base and diag:
        return f"{base}#{diag}"
    if base:
        return base
    if path:
        return path
    return "__aggregate__"

def collect_failure_cases(out_dir: Path, summary_rows):
    result_sources = [
        ("sv-tests", "BMC", out_dir / "sv-tests-bmc-results.txt"),
        ("sv-tests", "LEC", out_dir / "sv-tests-lec-results.txt"),
        ("verilator-verification", "BMC", out_dir / "verilator-bmc-results.txt"),
        ("verilator-verification", "LEC", out_dir / "verilator-lec-results.txt"),
        ("yosys/tests/sva", "BMC", out_dir / "yosys-bmc-results.txt"),
        ("yosys/tests/sva", "LEC", out_dir / "yosys-lec-results.txt"),
        ("opentitan", "LEC", out_dir / "opentitan-lec-results.txt"),
        ("opentitan", "LEC_STRICT", out_dir / "opentitan-lec-strict-results.txt"),
        ("opentitan", "E2E", out_dir / "opentitan-e2e-results.txt"),
        ("opentitan", "E2E_STRICT", out_dir / "opentitan-e2e-strict-results.txt"),
        ("opentitan", "E2E_MODE_DIFF", out_dir / "opentitan-e2e-mode-diff-results.txt"),
        ("", "", out_dir / "avip-results.txt"),
    ]
    detailed_pairs_observed = set()
    cases = {}
    for default_suite, default_mode, path in result_sources:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                status = parts[0].strip().upper() if parts else ""
                if status not in fail_like_statuses:
                    continue
                suite = parts[3].strip() if len(parts) > 3 and parts[3].strip() else default_suite
                mode = parts[4].strip() if len(parts) > 4 and parts[4].strip() else default_mode
                if not suite or not mode:
                    continue
                base = parts[1].strip() if len(parts) > 1 else ""
                file_path = parts[2].strip() if len(parts) > 2 else ""
                key = (suite, mode)
                detailed_pairs_observed.add(key)
                cases.setdefault(key, set()).add(compose_case_id(base, file_path))
    for row in summary_rows:
        suite = row.get("suite", "")
        mode = row.get("mode", "")
        if not suite or not mode:
            continue
        key = (suite, mode)
        if key in detailed_pairs_observed:
            continue
        try:
            fail = int(row.get("fail", "0"))
        except Exception:
            fail = 0
        try:
            error = int(row.get("error", "0"))
        except Exception:
            error = 0
        try:
            xfail = int(row.get("xfail", "0"))
        except Exception:
            xfail = 0
        try:
            xpass = int(row.get("xpass", "0"))
        except Exception:
            xpass = 0
        if fail > 0 or error > 0 or xfail > 0 or xpass > 0:
            cases.setdefault(key, set()).add("__aggregate__")
    return {key: ";".join(sorted(values)) for key, values in cases.items()}

def read_baseline_int(row, key, summary_counts):
    raw = row.get(key)
    if raw is not None and raw != "":
        try:
            return int(raw)
        except ValueError:
            pass
    return int(summary_counts.get(key, 0))

def read_int(row, key, fallback=0):
    value = row.get(key)
    if value is None or value == "":
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback

def compute_pass_rate(total: int, passed: int, xfail: int, skipped: int) -> float:
    eligible = total - skipped
    if eligible <= 0:
        return 0.0
    return ((passed + xfail) * 100.0) / eligible

failure_cases = collect_failure_cases(out_dir, rows)

baseline = {}
if baseline_path.exists():
    with baseline_path.open() as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            key = (row['date'], row['suite'], row['mode'])
            summary_counts = parse_result_summary(row.get('result', ''))
            total = read_int(row, 'total', summary_counts.get('total', 0))
            passed = read_int(row, 'pass', summary_counts.get('pass', 0))
            fail = read_int(row, 'fail', summary_counts.get('fail', 0))
            xfail = read_int(row, 'xfail', summary_counts.get('xfail', 0))
            xpass = read_int(row, 'xpass', summary_counts.get('xpass', 0))
            error = read_int(row, 'error', summary_counts.get('error', 0))
            skip = read_int(row, 'skip', summary_counts.get('skip', 0))
            pass_rate = row.get('pass_rate', '')
            if not pass_rate:
                pass_rate = f"{compute_pass_rate(total, passed, xfail, skip):.3f}"
            baseline[key] = {
                'date': row['date'],
                'suite': row['suite'],
                'mode': row['mode'],
                'total': str(total),
                'pass': str(passed),
                'fail': str(fail),
                'xfail': str(xfail),
                'xpass': str(xpass),
                'error': str(error),
                'skip': str(skip),
                'pass_rate': pass_rate,
                'result': row.get('result', ''),
                'failure_cases': row.get('failure_cases', ''),
            }

for row in rows:
    key = (date_str, row['suite'], row['mode'])
    total = int(row['total'])
    passed = int(row['pass'])
    fail = int(row['fail'])
    xfail = int(row['xfail'])
    xpass = int(row['xpass'])
    error = int(row['error'])
    skip = int(row['skip'])
    pass_rate = compute_pass_rate(total, passed, xfail, skip)
    baseline[key] = {
        'date': date_str,
        'suite': row['suite'],
        'mode': row['mode'],
        'total': str(total),
        'pass': str(passed),
        'fail': str(fail),
        'xfail': str(xfail),
        'xpass': str(xpass),
        'error': str(error),
        'skip': str(skip),
        'pass_rate': f"{pass_rate:.3f}",
        'result': row['summary'],
        'failure_cases': failure_cases.get((row['suite'], row['mode']), ''),
    }

baseline_path.parent.mkdir(parents=True, exist_ok=True)
with baseline_path.open('w', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            'date',
            'suite',
            'mode',
            'total',
            'pass',
            'fail',
            'xfail',
            'xpass',
            'error',
            'skip',
            'pass_rate',
            'result',
            'failure_cases',
        ],
        delimiter='\t',
    )
    writer.writeheader()
    for key in sorted(baseline.keys()):
        writer.writerow(baseline[key])

if plan_path.exists():
    data = plan_path.read_text().splitlines()
    header_idx = None
    table_end = None
    for idx, line in enumerate(data):
        if line.strip() == "| Date | Suite | Mode | Result | Notes |":
            header_idx = idx
            continue
        if header_idx is not None and idx > header_idx:
            if line.strip().startswith("|"):
                table_end = idx
                continue
            table_end = idx - 1
            break
    if header_idx is not None:
        if table_end is None:
            table_end = header_idx
        seen = set()
        for idx in range(header_idx + 1, table_end + 1):
            line = data[idx]
            parts = [p.strip() for p in line.strip().strip('|').split('|')]
            if len(parts) < 5:
                continue
            date, suite, mode, result, notes = parts[:5]
            key = (date, suite, mode)
            seen.add(key)
            if key in baseline:
                result = baseline[key]['result']
                data[idx] = f"| {date} | {suite} | {mode} | {result} | {notes} |"
        new_rows = []
        for key, row in sorted(baseline.items()):
            if row['date'] != date_str:
                continue
            if key in seen:
                continue
            new_rows.append(
                f"| {row['date']} | {row['suite']} | {row['mode']} | {row['result']} | added by script |"
            )
        if new_rows:
            insert_at = table_end + 1
            data[insert_at:insert_at] = new_rows
        plan_path.write_text("\n".join(data) + "\n")
PY
fi

if [[ "$FAIL_ON_NEW_XPASS" == "1" || \
      "$FAIL_ON_PASSRATE_REGRESSION" == "1" || \
      "$FAIL_ON_NEW_FAILURE_CASES" == "1" || \
      "$FAIL_ON_NEW_BMC_TIMEOUT_CASES" == "1" || \
      "$FAIL_ON_NEW_BMC_UNKNOWN_CASES" == "1" || \
      "$FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_FAIL" == "1" || \
      "$FAIL_ON_NEW_E2E_MODE_DIFF_STATUS_DIFF" == "1" || \
      "$FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_PASS" == "1" || \
      "$FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E" == "1" || \
      "$FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E_STRICT" == "1" || \
      -n "$OPENTITAN_LEC_STRICT_XPROP_COUNTER_KEYS_CSV" ]]; then
  OUT_DIR="$OUT_DIR" BASELINE_FILE="$BASELINE_FILE" \
  BASELINE_WINDOW="$BASELINE_WINDOW" \
  BASELINE_WINDOW_DAYS="$BASELINE_WINDOW_DAYS" \
  FAIL_ON_NEW_XPASS="$FAIL_ON_NEW_XPASS" \
  FAIL_ON_PASSRATE_REGRESSION="$FAIL_ON_PASSRATE_REGRESSION" \
  FAIL_ON_NEW_FAILURE_CASES="$FAIL_ON_NEW_FAILURE_CASES" \
  FAIL_ON_NEW_BMC_TIMEOUT_CASES="$FAIL_ON_NEW_BMC_TIMEOUT_CASES" \
  FAIL_ON_NEW_BMC_UNKNOWN_CASES="$FAIL_ON_NEW_BMC_UNKNOWN_CASES" \
  FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_FAIL="$FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_FAIL" \
  FAIL_ON_NEW_E2E_MODE_DIFF_STATUS_DIFF="$FAIL_ON_NEW_E2E_MODE_DIFF_STATUS_DIFF" \
  FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_PASS="$FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_PASS" \
  FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E="$FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E" \
  FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E_STRICT="$FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E_STRICT" \
  OPENTITAN_LEC_STRICT_XPROP_COUNTER_KEYS="$OPENTITAN_LEC_STRICT_XPROP_COUNTER_KEYS_CSV" \
  STRICT_GATE="$STRICT_GATE" python3 - <<'PY'
import csv
import datetime as dt
import os
import re
from pathlib import Path

summary_path = Path(os.environ["OUT_DIR"]) / "summary.tsv"
baseline_path = Path(os.environ["BASELINE_FILE"])

if not baseline_path.exists():
    raise SystemExit(f"baseline file not found: {baseline_path}")

def parse_result_summary(summary: str):
    parsed = {}
    for match in re.finditer(r"([a-z][a-z0-9_]*)=([0-9]+)", summary):
        parsed[match.group(1)] = int(match.group(2))
    return parsed

def read_baseline_int(row, key, summary_counts):
    raw = row.get(key)
    if raw is not None and raw != "":
        try:
            return int(raw)
        except ValueError:
            pass
    return int(summary_counts.get(key, 0))

def pass_rate(row):
    total = int(row.get("total", 0))
    passed = int(row.get("pass", 0))
    xfail = int(row.get("xfail", 0))
    skipped = int(row.get("skip", 0))
    eligible = total - skipped
    if eligible <= 0:
        return 0.0
    return ((passed + xfail) * 100.0) / eligible

fail_like_statuses = {"FAIL", "ERROR", "XFAIL", "XPASS", "EFAIL", "TIMEOUT", "UNKNOWN"}

def extract_diag_tag(path: str) -> str:
    if "#" not in path:
        return ""
    candidate = path.rsplit("#", 1)[1].strip()
    if re.fullmatch(r"[A-Z0-9_]+", candidate):
        return candidate
    return ""

def compose_case_id(base: str, path: str) -> str:
    diag = extract_diag_tag(path)
    if base and diag:
        return f"{base}#{diag}"
    if base:
        return base
    if path:
        return path
    return "__aggregate__"

def collect_failure_cases(out_dir: Path, summary_rows):
    result_sources = [
        ("sv-tests", "BMC", out_dir / "sv-tests-bmc-results.txt"),
        ("sv-tests", "LEC", out_dir / "sv-tests-lec-results.txt"),
        ("verilator-verification", "BMC", out_dir / "verilator-bmc-results.txt"),
        ("verilator-verification", "LEC", out_dir / "verilator-lec-results.txt"),
        ("yosys/tests/sva", "BMC", out_dir / "yosys-bmc-results.txt"),
        ("yosys/tests/sva", "LEC", out_dir / "yosys-lec-results.txt"),
        ("opentitan", "LEC", out_dir / "opentitan-lec-results.txt"),
        ("opentitan", "LEC_STRICT", out_dir / "opentitan-lec-strict-results.txt"),
        ("opentitan", "E2E", out_dir / "opentitan-e2e-results.txt"),
        ("opentitan", "E2E_STRICT", out_dir / "opentitan-e2e-strict-results.txt"),
        ("opentitan", "E2E_MODE_DIFF", out_dir / "opentitan-e2e-mode-diff-results.txt"),
        ("", "", out_dir / "avip-results.txt"),
    ]
    detailed_pairs_observed = set()
    cases = {}
    for default_suite, default_mode, path in result_sources:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                status = parts[0].strip().upper() if parts else ""
                if status not in fail_like_statuses:
                    continue
                suite = parts[3].strip() if len(parts) > 3 and parts[3].strip() else default_suite
                mode = parts[4].strip() if len(parts) > 4 and parts[4].strip() else default_mode
                if not suite or not mode:
                    continue
                base = parts[1].strip() if len(parts) > 1 else ""
                file_path = parts[2].strip() if len(parts) > 2 else ""
                key = (suite, mode)
                detailed_pairs_observed.add(key)
                cases.setdefault(key, set()).add(compose_case_id(base, file_path))
    for key, row in summary_rows.items():
        if key in detailed_pairs_observed:
            continue
        fail = int(row.get("fail", "0"))
        error = int(row.get("error", "0"))
        xfail = int(row.get("xfail", "0"))
        xpass = int(row.get("xpass", "0"))
        if fail > 0 or error > 0 or xfail > 0 or xpass > 0:
            cases.setdefault(key, set()).add("__aggregate__")
    return cases

summary = {}
with summary_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        key = (row["suite"], row["mode"])
        summary[key] = row

current_failure_cases = collect_failure_cases(Path(os.environ["OUT_DIR"]), summary)

history = {}
with baseline_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        suite = row.get("suite", "")
        mode = row.get("mode", "")
        if not suite or not mode:
            continue
        key = (suite, mode)
        history.setdefault(key, []).append(row)

fail_on_new_xpass = os.environ.get("FAIL_ON_NEW_XPASS", "0") == "1"
fail_on_passrate_regression = os.environ.get("FAIL_ON_PASSRATE_REGRESSION", "0") == "1"
fail_on_new_failure_cases = os.environ.get("FAIL_ON_NEW_FAILURE_CASES", "0") == "1"
fail_on_new_bmc_timeout_cases = (
    os.environ.get("FAIL_ON_NEW_BMC_TIMEOUT_CASES", "0") == "1"
)
fail_on_new_bmc_unknown_cases = (
    os.environ.get("FAIL_ON_NEW_BMC_UNKNOWN_CASES", "0") == "1"
)
fail_on_new_e2e_mode_diff_strict_only_fail = (
    os.environ.get("FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_FAIL", "0") == "1"
)
fail_on_new_e2e_mode_diff_status_diff = (
    os.environ.get("FAIL_ON_NEW_E2E_MODE_DIFF_STATUS_DIFF", "0") == "1"
)
fail_on_new_e2e_mode_diff_strict_only_pass = (
    os.environ.get("FAIL_ON_NEW_E2E_MODE_DIFF_STRICT_ONLY_PASS", "0") == "1"
)
fail_on_new_e2e_mode_diff_missing_in_e2e = (
    os.environ.get("FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E", "0") == "1"
)
fail_on_new_e2e_mode_diff_missing_in_e2e_strict = (
    os.environ.get("FAIL_ON_NEW_E2E_MODE_DIFF_MISSING_IN_E2E_STRICT", "0") == "1"
)
opentitan_lec_strict_xprop_counter_keys = [
    token.strip()
    for token in os.environ.get("OPENTITAN_LEC_STRICT_XPROP_COUNTER_KEYS", "").split(",")
    if token.strip()
]
strict_gate = os.environ.get("STRICT_GATE", "0") == "1"
baseline_window = int(os.environ.get("BASELINE_WINDOW", "1"))
baseline_window_days = int(os.environ.get("BASELINE_WINDOW_DAYS", "0"))

gate_errors = []
for key, current_row in summary.items():
    suite, mode = key
    history_rows = history.get(key, [])
    if not history_rows:
        if strict_gate:
            gate_errors.append(f"{suite} {mode}: missing baseline row")
        continue
    history_rows.sort(key=lambda r: r.get("date", ""))
    if baseline_window_days > 0:
        parsed_dates = []
        for row in history_rows:
            try:
                parsed_dates.append(dt.date.fromisoformat(row.get("date", "")))
            except Exception:
                parsed_dates.append(None)
        valid_dates = [d for d in parsed_dates if d is not None]
        if valid_dates:
            latest_date = max(valid_dates)
            cutoff = latest_date - dt.timedelta(days=baseline_window_days)
            filtered_rows = []
            for row, row_date in zip(history_rows, parsed_dates):
                if row_date is None:
                    continue
                if cutoff <= row_date <= latest_date:
                    filtered_rows.append(row)
            history_rows = filtered_rows
    if strict_gate and len(history_rows) < baseline_window:
        gate_errors.append(
            f"{suite} {mode}: insufficient baseline history ({len(history_rows)} < {baseline_window})"
        )
        continue
    if not history_rows:
        if strict_gate:
            gate_errors.append(
                f"{suite} {mode}: no baseline rows remain after baseline-window-days={baseline_window_days} filtering"
            )
        continue
    compare_rows = history_rows[-baseline_window:]
    parsed_counts = [parse_result_summary(row.get("result", "")) for row in compare_rows]
    baseline_fail = min(
        read_baseline_int(row, "fail", counts)
        for row, counts in zip(compare_rows, parsed_counts)
    )
    baseline_error = min(
        read_baseline_int(row, "error", counts)
        for row, counts in zip(compare_rows, parsed_counts)
    )
    baseline_xpass = min(
        read_baseline_int(row, "xpass", counts)
        for row, counts in zip(compare_rows, parsed_counts)
    )
    current_fail = int(current_row["fail"])
    current_error = int(current_row["error"])
    current_xpass = int(current_row["xpass"])
    if current_fail > baseline_fail:
        gate_errors.append(
            f"{suite} {mode}: fail increased ({baseline_fail} -> {current_fail}, window={baseline_window})"
        )
    if current_error > baseline_error:
        gate_errors.append(
            f"{suite} {mode}: error increased ({baseline_error} -> {current_error}, window={baseline_window})"
        )
    if fail_on_new_xpass and current_xpass > baseline_xpass:
        gate_errors.append(
            f"{suite} {mode}: xpass increased ({baseline_xpass} -> {current_xpass}, window={baseline_window})"
        )
    if fail_on_new_failure_cases:
        baseline_cases_raw = [row.get("failure_cases") for row in compare_rows]
        # Legacy baseline rows may not carry case telemetry; skip this check there.
        if any(raw is not None for raw in baseline_cases_raw):
            baseline_case_set = set()
            for raw in baseline_cases_raw:
                if raw is None or raw == "":
                    continue
                for token in raw.split(";"):
                    token = token.strip()
                    if token:
                        baseline_case_set.add(token)
            current_case_set = current_failure_cases.get(key, set())
            new_cases = sorted(current_case_set - baseline_case_set)
            if new_cases:
                sample = ", ".join(new_cases[:3])
                if len(new_cases) > 3:
                    sample += ", ..."
                gate_errors.append(
                    f"{suite} {mode}: new failure cases observed (baseline={len(baseline_case_set)} current={len(current_case_set)}, window={baseline_window}): {sample}"
                )
    if mode == "BMC":
        current_counts = parse_result_summary(current_row.get("summary", ""))
        if fail_on_new_bmc_timeout_cases:
            baseline_timeout_values = []
            for counts in parsed_counts:
                if "bmc_timeout_cases" in counts:
                    baseline_timeout_values.append(int(counts["bmc_timeout_cases"]))
            if baseline_timeout_values:
                baseline_timeout = min(baseline_timeout_values)
                current_timeout = int(current_counts.get("bmc_timeout_cases", 0))
                if current_timeout > baseline_timeout:
                    gate_errors.append(
                        f"{suite} {mode}: bmc_timeout_cases increased ({baseline_timeout} -> {current_timeout}, window={baseline_window})"
                    )
        if fail_on_new_bmc_unknown_cases:
            baseline_unknown_values = []
            for counts in parsed_counts:
                if "bmc_unknown_cases" in counts:
                    baseline_unknown_values.append(int(counts["bmc_unknown_cases"]))
            if baseline_unknown_values:
                baseline_unknown = min(baseline_unknown_values)
                current_unknown = int(current_counts.get("bmc_unknown_cases", 0))
                if current_unknown > baseline_unknown:
                    gate_errors.append(
                        f"{suite} {mode}: bmc_unknown_cases increased ({baseline_unknown} -> {current_unknown}, window={baseline_window})"
                    )
    if suite == "opentitan" and mode == "E2E_MODE_DIFF":
        current_counts = parse_result_summary(current_row.get("summary", ""))
        if fail_on_new_e2e_mode_diff_strict_only_fail:
            baseline_strict_only_fail_values = []
            for counts in parsed_counts:
                if "strict_only_fail" in counts:
                    baseline_strict_only_fail_values.append(
                        int(counts["strict_only_fail"])
                    )
            if baseline_strict_only_fail_values:
                baseline_strict_only_fail = min(baseline_strict_only_fail_values)
                current_strict_only_fail = int(
                    current_counts.get("strict_only_fail", 0)
                )
                if current_strict_only_fail > baseline_strict_only_fail:
                    gate_errors.append(
                        f"{suite} {mode}: strict_only_fail increased ({baseline_strict_only_fail} -> {current_strict_only_fail}, window={baseline_window})"
                    )
        if fail_on_new_e2e_mode_diff_status_diff:
            baseline_status_diff_values = []
            for counts in parsed_counts:
                if "status_diff" in counts:
                    baseline_status_diff_values.append(int(counts["status_diff"]))
            if baseline_status_diff_values:
                baseline_status_diff = min(baseline_status_diff_values)
                current_status_diff = int(current_counts.get("status_diff", 0))
                if current_status_diff > baseline_status_diff:
                    gate_errors.append(
                        f"{suite} {mode}: status_diff increased ({baseline_status_diff} -> {current_status_diff}, window={baseline_window})"
                    )
        if fail_on_new_e2e_mode_diff_strict_only_pass:
            baseline_strict_only_pass_values = []
            for counts in parsed_counts:
                if "strict_only_pass" in counts:
                    baseline_strict_only_pass_values.append(
                        int(counts["strict_only_pass"])
                    )
            if baseline_strict_only_pass_values:
                baseline_strict_only_pass = min(baseline_strict_only_pass_values)
                current_strict_only_pass = int(
                    current_counts.get("strict_only_pass", 0)
                )
                if current_strict_only_pass > baseline_strict_only_pass:
                    gate_errors.append(
                        f"{suite} {mode}: strict_only_pass increased ({baseline_strict_only_pass} -> {current_strict_only_pass}, window={baseline_window})"
                    )
        if fail_on_new_e2e_mode_diff_missing_in_e2e:
            baseline_missing_in_e2e_values = []
            for counts in parsed_counts:
                if "missing_in_e2e" in counts:
                    baseline_missing_in_e2e_values.append(
                        int(counts["missing_in_e2e"])
                    )
            if baseline_missing_in_e2e_values:
                baseline_missing_in_e2e = min(baseline_missing_in_e2e_values)
                current_missing_in_e2e = int(current_counts.get("missing_in_e2e", 0))
                if current_missing_in_e2e > baseline_missing_in_e2e:
                    gate_errors.append(
                        f"{suite} {mode}: missing_in_e2e increased ({baseline_missing_in_e2e} -> {current_missing_in_e2e}, window={baseline_window})"
                    )
        if fail_on_new_e2e_mode_diff_missing_in_e2e_strict:
            baseline_missing_in_e2e_strict_values = []
            for counts in parsed_counts:
                if "missing_in_e2e_strict" in counts:
                    baseline_missing_in_e2e_strict_values.append(
                        int(counts["missing_in_e2e_strict"])
                    )
            if baseline_missing_in_e2e_strict_values:
                baseline_missing_in_e2e_strict = min(
                    baseline_missing_in_e2e_strict_values
                )
                current_missing_in_e2e_strict = int(
                    current_counts.get("missing_in_e2e_strict", 0)
                )
                if current_missing_in_e2e_strict > baseline_missing_in_e2e_strict:
                    gate_errors.append(
                        f"{suite} {mode}: missing_in_e2e_strict increased ({baseline_missing_in_e2e_strict} -> {current_missing_in_e2e_strict}, window={baseline_window})"
                    )
    if suite == "opentitan" and mode == "LEC_STRICT" and opentitan_lec_strict_xprop_counter_keys:
        current_counts = parse_result_summary(current_row.get("summary", ""))
        for counter_key in opentitan_lec_strict_xprop_counter_keys:
            baseline_values = []
            for counts in parsed_counts:
                if counter_key in counts:
                    baseline_values.append(int(counts[counter_key]))
            if not baseline_values:
                continue
            baseline_counter = min(baseline_values)
            current_counter = int(current_counts.get(counter_key, 0))
            if current_counter > baseline_counter:
                gate_errors.append(
                    f"{suite} {mode}: {counter_key} increased ({baseline_counter} -> {current_counter}, window={baseline_window})"
                )
    if fail_on_passrate_regression:
        baseline_rate = max(
            pass_rate(
                {
                    "total": read_baseline_int(row, "total", counts),
                    "pass": read_baseline_int(row, "pass", counts),
                    "xfail": read_baseline_int(row, "xfail", counts),
                    "skip": read_baseline_int(row, "skip", counts),
                }
            )
            for row, counts in zip(compare_rows, parsed_counts)
        )
        current_rate = pass_rate(current_row)
        if current_rate + 1e-9 < baseline_rate:
            gate_errors.append(
                f"{suite} {mode}: pass_rate regressed ({baseline_rate:.3f} -> {current_rate:.3f}, window={baseline_window})"
            )

if gate_errors:
    print("strict gate failures:")
    for item in gate_errors:
        print(f"  {item}")
    raise SystemExit(1)
PY
fi

if [[ "$FAIL_ON_DIFF" == "1" ]]; then
  OUT_DIR="$OUT_DIR" BASELINE_FILE="$BASELINE_FILE" python3 - <<'PY'
import csv
import os
from pathlib import Path

summary_path = Path(os.environ["OUT_DIR"]) / "summary.tsv"
baseline_path = Path(os.environ["BASELINE_FILE"])

if not baseline_path.exists():
    raise SystemExit("baseline file not found: utils/formal-baselines.tsv")

summary = {}
with summary_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        key = (row['suite'], row['mode'])
        summary[key] = row['summary']

latest = {}
with baseline_path.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        key = (row['suite'], row['mode'])
        latest.setdefault(key, []).append(row)

for key, entries in latest.items():
    entries.sort(key=lambda r: r['date'])
    latest[key] = entries[-1]['result']

diffs = []
for key, summary_val in summary.items():
    baseline_val = latest.get(key)
    if baseline_val is None:
        diffs.append(f"missing baseline for {key[0]} {key[1]}")
    elif baseline_val != summary_val:
        diffs.append(f"{key[0]} {key[1]}: {baseline_val} -> {summary_val}")

if diffs:
    print("baseline diffs:")
    for item in diffs:
        print(f"  {item}")
    raise SystemExit(1)
PY
fi

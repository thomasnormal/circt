#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
REPO_ROOT="$(checks_repo_root "$SCRIPT_DIR")"
RUNNER="$REPO_ROOT/utils/run_avip_circt_sim.sh"

if [[ ! -x "$RUNNER" ]]; then
  echo "[avip-circt-sim-tool-canonical-check] missing executable: $RUNNER" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mkdir -p "$tmpdir/build_test/bin"
cat >"$tmpdir/build_test/bin/circt-verilog" <<'SH'
#!/usr/bin/env bash
exit 0
SH
cat >"$tmpdir/build_test/bin/circt-sim" <<'SH'
#!/usr/bin/env bash
exit 0
SH
chmod +x "$tmpdir/build_test/bin/circt-verilog" "$tmpdir/build_test/bin/circt-sim"

set +e
out="$tmpdir/run.out"
err="$tmpdir/run.err"
AVIPS=bogus \
RUN_AVIP=/bin/true \
CIRCT_ROOT="$tmpdir" \
MBIT_DIR="$tmpdir/mbit" \
  "$RUNNER" "$tmpdir/outdir" >"$out" 2>"$err"
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  echo "[avip-circt-sim-tool-canonical-check] expected non-zero exit for AVIPS=bogus" >&2
  exit 1
fi
if ! rg -q "no AVIPs selected" "$err"; then
  echo "[avip-circt-sim-tool-canonical-check] missing expected selection failure" >&2
  cat "$err" >&2
  exit 1
fi
if rg -q "circt-sim not found|circt-verilog not found" "$err"; then
  echo "[avip-circt-sim-tool-canonical-check] canonical tool resolution failed" >&2
  cat "$err" >&2
  exit 1
fi

echo "[avip-circt-sim-tool-canonical-check] PASS"

# Illegal SV Differential Corpus

This corpus is for differential front-end testing between Xcelium and CIRCT.

Each `.sv` file is intentionally illegal SystemVerilog and should be checked with:

- Xcelium (recommended): `xrun -sv <case> -elaborate`
- CIRCT (full import path): `circt-verilog --no-uvm-auto-include <case>`
- CIRCT (lint mode): `circt-verilog --no-uvm-auto-include --lint-only <case>`

Use `utils/run_illegal_sv_xcelium_diff.py` to run the full corpus and classify
results. The script defaults to CIRCT full mode; pass `--circt-mode lint-only`
to compare in lint mode.

Cases can include expected diagnostic tags:

- `// EXPECT_CIRCT_DIAG: <substring>`
- `// EXPECT_XCELIUM_DIAG: <substring>`

The differential script validates these substrings against tool output and can
fail on mismatches with `--fail-on-expect-mismatch`.

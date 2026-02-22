# Critical/High Fix Tracker

Date: 2026-02-20

Scope: fix critical/high issues identified in codebase audit.

- [x] JIT block-compiler lifetime safety:
  keep compiled function pointers valid across multiple block compiles.
- [x] JIT delay encoding/scheduling correctness:
  remove signed bit-decoding hazards; honor epsilon in deferred scheduling.
- [x] Arcilator BehavioralLowering fail-fast:
  replace semantic no-op/skeleton lowerings with explicit unsupported errors.
- [x] Unified regression runner signal handling:
  terminate child lane processes cleanly on INT/TERM.
- [x] Unified regression runner CLI robustness:
  emit clear "missing value" errors for options that require arguments.
- [x] Add/update focused tests for the above changes.
- [x] Run focused validation (lit/unit/build) and record outcomes.

Validation log:

- `bash -n utils/run_regression_unified.sh` (PASS)
- `python3 -m compileall -q utils test/Tools/Inputs` (PASS)
- `ninja -C build-test circt-sim arcilator CIRCTSimToolTests` (PASS)
- `build-test/unittests/Tools/circt-sim/CIRCTSimToolTests --gtest_filter='*JITDelay*'` (PASS)
- `build-test/bin/arcilator test/arcilator/arcilator.mlir` (PASS)
- `build-test/bin/arcilator test/arcilator/compreg.mlir` (PASS)
- `build-ot/bin/llvm-lit -sv test/Tools --filter='run-regression-unified-.*|formal-runner-common-retry'` (PASS)

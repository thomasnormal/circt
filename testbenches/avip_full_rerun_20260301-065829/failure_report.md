# AVIP Full Rerun Failure Report (2026-03-01)

- Artifact base: `testbenches/avip_full_rerun_20260301-065829`
- Lanes: `interpret`, `compile`
- AVIPs: `all9`
- Seed: `1`

## Interpret Summary

- compile OK: `5/9`
- sim OK: `0/9`

| AVIP | Compile | Sim | Failure Signature |
|---|---|---|---|
| apb | OK (18s) | FAIL (exit=0, 4s) | UVM_FATAL @                 0 ns: reporter [INVTST] Requested test from command line +UVM_TESTNAME=apb_8b_write_test not found. \| [circt-sim] Main loop exit: shouldContinue()=false at time 0 fs, iter=1, deltas=3 \| [circt-sim] Simulation terminated at time 0 fs (success=true, verbose=false) |
| ahb | OK (19s) | FAIL (exit=0, 4s) | UVM_FATAL @                 0 ns: reporter [INVTST] Requested test from command line +UVM_TESTNAME=AhbWriteTest not found. \| [circt-sim] Main loop exit: shouldContinue()=false at time 0 fs, iter=1, deltas=3 \| [circt-sim] Simulation terminated at time 0 fs (success=true, verbose=false) |
| axi4 | FAIL (14s) | SKIP (exit=-, -s) | zero-byte MLIR output |
| axi4Lite | OK (64s) | FAIL (exit=0, 6s) | UVM_FATAL @                 0 ns: reporter [INVTST] Requested test from command line +UVM_TESTNAME=Axi4LiteWriteTest not found. \| [circt-sim] Main loop exit: shouldContinue()=false at time 0 fs, iter=1, deltas=3 \| [circt-sim] Simulation terminated at time 0 fs (success=true, verbose=false) |
| i2s | OK (29s) | FAIL (exit=0, 4s) | UVM_FATAL @                 0 ns: reporter [INVTST] Requested test from command line +UVM_TESTNAME=I2sWriteOperationWith8bitdataTxMasterRxSlaveWith48khzTest not found. \| [circt-sim] Main loop exit: shouldContinue()=false at time 0 fs, iter=1, deltas=3 \| [circt-sim] Simulation terminated at time 0 fs (success=true, verbose=false) |
| i3c | OK (23s) | FAIL (exit=0, 4s) | UVM_FATAL @                 0 ns: reporter [INVTST] Requested test from command line +UVM_TESTNAME=i3c_writeOperationWith8bitsData_test not found. \| [circt-sim] Main loop exit: shouldContinue()=false at time 0 fs, iter=1, deltas=3 \| [circt-sim] Simulation terminated at time 0 fs (success=true, verbose=false) |
| jtag | FAIL (0s) | SKIP (exit=-, -s) | zero-byte MLIR output |
| spi | FAIL (0s) | SKIP (exit=-, -s) | zero-byte MLIR output |
| uart | FAIL (0s) | SKIP (exit=-, -s) | zero-byte MLIR output |

## Compile Summary

- compile OK: `5/9`
- sim OK: `0/9`

| AVIP | Compile | Sim | Failure Signature |
|---|---|---|---|
| apb | OK (88s) | FAIL (exit=139, 4s) | UVM_FATAL @                 0 ns: reporter [INVTST] Requested test from command line +UVM_TESTNAME=apb_8b_write_test not found. \| utils/run_avip_circt_sim.sh: line 343: 1279007 Segmentation fault      timeout --signal=KILL "$timeout_secs" "$@" |
| ahb | OK (87s) | FAIL (exit=139, 4s) | UVM_FATAL @                 0 ns: reporter [INVTST] Requested test from command line +UVM_TESTNAME=AhbWriteTest not found. \| utils/run_avip_circt_sim.sh: line 343: 1283923 Segmentation fault      timeout --signal=KILL "$timeout_secs" "$@" |
| axi4 | FAIL (15s) | SKIP (exit=-, -s) | zero-byte MLIR output |
| axi4Lite | OK (207s) | FAIL (exit=139, 9s) | UVM_FATAL @                 0 ns: reporter [INVTST] Requested test from command line +UVM_TESTNAME=Axi4LiteWriteTest not found. \| utils/run_avip_circt_sim.sh: line 343: 1315282 Segmentation fault      timeout --signal=KILL "$timeout_secs" "$@" |
| i2s | OK (116s) | FAIL (exit=139, 4s) | UVM_FATAL @                 0 ns: reporter [INVTST] Requested test from command line +UVM_TESTNAME=I2sWriteOperationWith8bitdataTxMasterRxSlaveWith48khzTest not found. \| utils/run_avip_circt_sim.sh: line 343: 1328340 Segmentation fault      timeout --signal=KILL "$timeout_secs" "$@" |
| i3c | OK (95s) | FAIL (exit=139, 5s) | UVM_FATAL @                 0 ns: reporter [INVTST] Requested test from command line +UVM_TESTNAME=i3c_writeOperationWith8bitsData_test not found. \| utils/run_avip_circt_sim.sh: line 343: 1334350 Segmentation fault      timeout --signal=KILL "$timeout_secs" "$@" |
| jtag | FAIL (0s) | SKIP (exit=-, -s) | zero-byte MLIR output |
| spi | FAIL (0s) | SKIP (exit=-, -s) | zero-byte MLIR output |
| uart | FAIL (0s) | SKIP (exit=-, -s) | zero-byte MLIR output |

## Failures To Investigate

1. Frontend compile emits zero-byte MLIR for `axi4`, `jtag`, `spi`, `uart` in both lanes.
2. All sim-started AVIPs fail with `UVM_FATAL [INVTST] Requested test from command line +UVM_TESTNAME=... not found` at `0 ns`.
3. Compile lane additionally exits with `sim_exit=139` after the same UVM fatal in all sim-started AVIPs.

Raw machine-readable table: `failure_summary.tsv`.

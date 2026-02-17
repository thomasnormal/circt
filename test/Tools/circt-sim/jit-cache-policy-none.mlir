// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=2 --jit-cache-policy=none --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit_config":
// JSON: "cache_policy": "none"
// JSON: "scheduler":
// JSON: "processes_executed": 2
// JSON: "jit":
// JSON: "jit_compiles_total": 2
// JSON: "jit_cache_hits_total": 2
// JSON: "jit_exec_hits_total": 2
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0
// JSON: "jit_strict_violations_total": 0

hw.module @top() {
  %delay = llhd.constant_time <1ns, 0d, 0e>
  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.halt
  }
  hw.output
}

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=7 --jit-compile-budget=11 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 1
// JSON: "jit_deopt_reason_missing_thunk": 1
// JSON: "jit_deopt_processes": [
// JSON: "reason": "missing_thunk"
// JSON: "detail": "below_hot_threshold"

hw.module @top() {
  %fmt = sim.fmt.literal "jit-report-missing-thunk-hot-threshold\0A"

  llhd.process {
    sim.proc.print %fmt
    llhd.halt
  }

  hw.output
}

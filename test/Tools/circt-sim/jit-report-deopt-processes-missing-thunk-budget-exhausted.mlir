// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_compiles_total": 1
// JSON: "jit_deopts_total": 1
// JSON: "jit_deopt_reason_missing_thunk": 1
// JSON: "jit_deopt_processes": [
// JSON: "reason": "missing_thunk"
// JSON: "detail": "compile_budget_exhausted"

hw.module @top() {
  %fmt0 = sim.fmt.literal "jit-report-missing-thunk-budget-exhausted-0\0A"
  %fmt1 = sim.fmt.literal "jit-report-missing-thunk-budget-exhausted-1\0A"

  llhd.process {
    sim.proc.print %fmt0
    llhd.halt
  }

  llhd.process {
    sim.proc.print %fmt1
    llhd.halt
  }

  hw.output
}

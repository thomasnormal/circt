// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: env CIRCT_SIM_JIT_FAIL_ON_DEOPT=1 not circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=0 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: [circt-sim] Strict JIT policy violation: deopts_total=1
// LOG: [circt-sim] Strict JIT deopt process: id={{[1-9][0-9]*}} name=llhd_process_{{[0-9]+}} reason=missing_thunk detail=compile_budget_zero
// LOG: [circt-sim] Simulation finished with exit code 1
//
// JSON: "mode": "compile"
// JSON: "jit_config":
// JSON: "fail_on_deopt": 1
// JSON: "jit":
// JSON: "jit_deopts_total": 1
// JSON: "jit_deopt_reason_missing_thunk": 1
// JSON: "jit_strict_violations_total": 1
// JSON: "jit_deopt_processes": [
// JSON: "reason": "missing_thunk"
// JSON: "detail": "compile_budget_zero"

hw.module @top() {
  %fmt = sim.fmt.literal "jit-fail-on-deopt-missing-thunk-budget-zero-detail\0A"

  llhd.process {
    sim.proc.print %fmt
    llhd.halt
  }

  hw.output
}

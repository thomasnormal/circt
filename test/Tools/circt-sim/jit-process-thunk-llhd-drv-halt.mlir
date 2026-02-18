// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: llhd_drv_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

hw.module @top() {
  %false = hw.constant false
  %true = hw.constant true
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %fmt = sim.fmt.literal "llhd_drv_ok\0A"

  %sig = llhd.sig %false : i1

  llhd.process {
    llhd.drv %sig, %true after %eps : i1
    sim.proc.print %fmt
    llhd.halt
  }

  hw.output
}

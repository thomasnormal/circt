// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=2 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_processes": []

hw.module @top() {
  %fmt = sim.fmt.literal "jit-report-deopt-processes\0A"
  %delay = llhd.constant_time <1ns, 0d, 0e>

  // Lock that deopt-process reporting remains empty for this single-process
  // compile-mode run.
  llhd.process {
    sim.proc.print %fmt
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.halt
  }

  hw.output
}

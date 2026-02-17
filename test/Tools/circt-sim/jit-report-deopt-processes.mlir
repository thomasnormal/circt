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
// JSON: "jit_deopts_total": 1
// JSON: "jit_deopt_reason_unsupported_operation": 1
// JSON: "jit_deopt_processes": [
// JSON: "process_id": {{[1-9][0-9]*}}
// JSON: "process_name": "llhd_process_{{[0-9]+}}"
// JSON: "reason": "unsupported_operation"
// JSON: "detail": "prewait_impure:sim.proc.print"

hw.module @top() {
  %fmt = sim.fmt.literal "jit-report-deopt-processes\0A"
  %delay = llhd.constant_time <1ns, 0d, 0e>

  // This process shape is currently not eligible for native-thunk install:
  // side-effecting pre-wait op with no results (`sim.proc.print`) in entry.
  llhd.process {
    sim.proc.print %fmt
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.halt
  }

  hw.output
}

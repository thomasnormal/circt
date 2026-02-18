// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=4 --jit-fail-on-deopt --max-time=10 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: parent_done
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  hw.module @top() {
    %fmt = sim.fmt.literal "parent_done\0A"

    llhd.process {
      %h = sim.fork join_type "join_none" {
        %child = sim.fork join_type "join_none" {
          sim.fork.terminator
        }
        sim.disable_fork
        sim.fork.terminator
      }

      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

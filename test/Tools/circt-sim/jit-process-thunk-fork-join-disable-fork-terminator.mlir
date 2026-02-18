// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=16 --jit-fail-on-deopt --max-time=10 --jit-report=%t/jit.json > %t/log.txt 2>&1
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
  llvm.func @__moore_wait_condition(i32)

  hw.module @top() {
    %zero = hw.constant 0 : i32
    %fmt = sim.fmt.literal "parent_done\0A"

    llhd.process {
      %h = sim.fork {
        %inner = sim.fork join_type "join_any" {
          llvm.call @__moore_wait_condition(%zero) : (i32) -> ()
          sim.fork.terminator
        }, {
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

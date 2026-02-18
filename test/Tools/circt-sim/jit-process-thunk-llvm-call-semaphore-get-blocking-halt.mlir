// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=16 --jit-fail-on-deopt --max-time=5000000000 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG-DAG: semaphore_put
// LOG-DAG: semaphore_got
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  llvm.func @__moore_semaphore_get(i64, i32) -> ()
  llvm.func @__moore_semaphore_put(i64, i32) -> ()

  hw.module @top() {
    %sem = hw.constant 4096 : i64
    %one = hw.constant 1 : i32
    %delay = llhd.constant_time <1ns, 0d, 0e>
    %got_fmt = sim.fmt.literal "semaphore_got\0A"
    %put_fmt = sim.fmt.literal "semaphore_put\0A"

    llhd.process {
      llvm.call @__moore_semaphore_get(%sem, %one) : (i64, i32) -> ()
      sim.proc.print %got_fmt
      llhd.halt
    }

    llhd.process {
      llhd.wait delay %delay, ^bb1
    ^bb1:
      llvm.call @__moore_semaphore_put(%sem, %one) : (i64, i32) -> ()
      sim.proc.print %put_fmt
      llhd.halt
    }

    hw.output
  }
}

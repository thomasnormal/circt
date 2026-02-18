// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: dyn_cast_check_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  llvm.func @__moore_dyn_cast_check(i32, i32, i32) -> i1

  hw.module @top() {
    %zero = hw.constant 0 : i32
    %one = hw.constant 1 : i32
    %fmt = sim.fmt.literal "dyn_cast_check_ok\0A"

    llhd.process {
      %ok = llvm.call @__moore_dyn_cast_check(%zero, %zero, %one)
          : (i32, i32, i32) -> i1
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

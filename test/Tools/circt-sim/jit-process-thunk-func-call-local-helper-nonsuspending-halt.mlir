// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --parallel=4 --work-stealing --auto-partition --max-time=2500 --jit-report=%t/jit-parallel.json > %t/log-parallel.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
// RUN: FileCheck %s --check-prefix=LOG < %t/log-parallel.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit-parallel.json
//
// LOG: local_helper_nonsuspending_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func private @local_helper_nonsuspending(%arg0: i32) -> i32 {
    %one = arith.constant 1 : i32
    %sum = arith.addi %arg0, %one : i32
    return %sum : i32
  }

  hw.module @top() {
    %fmt = sim.fmt.literal "local_helper_nonsuspending_ok\0A"
    %seed = arith.constant 41 : i32

    llhd.process {
      %result = func.call @local_helper_nonsuspending(%seed) : (i32) -> i32
      %unused = comb.icmp uge %result, %result : i32
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

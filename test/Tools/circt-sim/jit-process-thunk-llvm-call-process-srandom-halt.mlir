// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: srandom_done
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  llvm.func @__moore_process_self() -> !llvm.ptr
  llvm.func @__moore_process_srandom(!llvm.ptr, i32)

  hw.module @top() {
    %seed = hw.constant 123 : i32
    %fmt = sim.fmt.literal "srandom_done\0A"

    llhd.process {
      %self = llvm.call @__moore_process_self() : () -> !llvm.ptr
      llvm.call @__moore_process_srandom(%self, %seed) : (!llvm.ptr, i32) -> ()
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: queue_push_back_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  llvm.func @__moore_queue_push_back(!llvm.ptr, !llvm.ptr, i64)

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %one = hw.constant 1 : i64
    %fmt = sim.fmt.literal "queue_push_back_ok\0A"

    llhd.process {
      llvm.call @__moore_queue_push_back(%null, %null, %one) : (!llvm.ptr, !llvm.ptr, i64) -> ()
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

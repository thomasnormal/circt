// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: queue_delete_index_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  llvm.func @__moore_queue_delete_index(!llvm.ptr, i32, i64)

  hw.module @top() {
    %fmt = sim.fmt.literal "queue_delete_index_ok\0A"
    %null = llvm.mlir.zero : !llvm.ptr
    %idx = hw.constant 0 : i32
    %size = hw.constant 1 : i64

    llhd.process {
      llvm.call @__moore_queue_delete_index(%null, %idx, %size) : (!llvm.ptr, i32, i64) -> ()
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

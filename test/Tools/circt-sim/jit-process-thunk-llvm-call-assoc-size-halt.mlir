// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: assoc_size_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  llvm.func @__moore_assoc_size(!llvm.ptr) -> i64

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %fmt = sim.fmt.literal "assoc_size_ok\0A"

    llhd.process {
      %size = llvm.call @__moore_assoc_size(%null) : (!llvm.ptr) -> i64
      %unused = arith.addi %size, %size : i64
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=8 --jit-fail-on-deopt --max-time=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: randomize_basic_done
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %zero = hw.constant 0 : i64
    %fmt = sim.fmt.literal "randomize_basic_done\0A"

    llhd.process {
      %ok = llvm.call @__moore_randomize_basic(%null, %zero) : (!llvm.ptr, i64) -> i32
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

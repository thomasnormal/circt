// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: uvm_root_find_all_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func @"uvm_pkg::uvm_root::find_all"(
      %arg0: !llvm.ptr, %arg1: !llvm.struct<(ptr, i64)>, %arg2: !llvm.ptr,
      %arg3: !llvm.ptr) {
    return
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %name = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %fmt = sim.fmt.literal "uvm_root_find_all_ok\0A"

    llhd.process {
      func.call @"uvm_pkg::uvm_root::find_all"(%null, %name, %null, %null) : (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.ptr, !llvm.ptr) -> ()
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: set_report_id_verbosity_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func @"uvm_pkg::uvm_report_object::set_report_id_verbosity"(
      %arg0: !llvm.ptr, %arg1: !llvm.struct<(ptr, i64)>, %arg2: i32) {
    return
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %id = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %level = hw.constant 7 : i32
    %fmt = sim.fmt.literal "set_report_id_verbosity_ok\0A"

    llhd.process {
      func.call @"uvm_pkg::uvm_report_object::set_report_id_verbosity"(%null, %id, %level) : (!llvm.ptr, !llvm.struct<(ptr, i64)>, i32) -> ()
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

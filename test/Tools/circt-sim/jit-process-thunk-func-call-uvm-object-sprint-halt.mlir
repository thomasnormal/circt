// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: uvm_object_sprint_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func private @"uvm_pkg::uvm_object::sprint"(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> !llvm.struct<(ptr, i64)> {
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %null = llvm.mlir.zero : !llvm.ptr
    %len = hw.constant 0 : i64
    %withPtr = llvm.insertvalue %null, %undef[0] : !llvm.struct<(ptr, i64)>
    %ret = llvm.insertvalue %len, %withPtr[1] : !llvm.struct<(ptr, i64)>
    return %ret : !llvm.struct<(ptr, i64)>
  }

  hw.module @top() {
    %fmt = sim.fmt.literal "uvm_object_sprint_ok\0A"
    %null = llvm.mlir.zero : !llvm.ptr

    llhd.process {
      %s = func.call @"uvm_pkg::uvm_object::sprint"(%null, %null) : (!llvm.ptr, !llvm.ptr) -> !llvm.struct<(ptr, i64)>
      %len = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>
      %unused = comb.icmp uge %len, %len : i64
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: configdb_set_byvalue_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func private @set_1234(%ctx: !llvm.ptr, %inst: !llvm.struct<(ptr, i64)>,
                              %field: !llvm.struct<(ptr, i64)>,
                              %value: i32) {
    return
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %zero64 = hw.constant 0 : i64
    %zero32 = hw.constant 0 : i32
    %fmt = sim.fmt.literal "configdb_set_byvalue_ok\0A"
    %emptyStr = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %emptyStr0 = llvm.insertvalue %null, %emptyStr[0] : !llvm.struct<(ptr, i64)>
    %emptyStr1 = llvm.insertvalue %zero64, %emptyStr0[1] : !llvm.struct<(ptr, i64)>

    llhd.process {
      func.call @set_1234(%null, %emptyStr1, %emptyStr1, %zero32) :
          (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, i32) -> ()
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

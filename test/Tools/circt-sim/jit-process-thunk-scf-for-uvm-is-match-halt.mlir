// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: scf_for_uvm_is_match_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func private @"uvm_pkg::uvm_is_match"(%arg0: !llvm.struct<(ptr, i64)>,
                                              %arg1: !llvm.struct<(ptr, i64)>) -> i1 {
    %true = hw.constant true
    return %true : i1
  }

  hw.module @top() {
    %fmt = sim.fmt.literal "scf_for_uvm_is_match_ok\0A"

    llhd.process {
      %undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
      %null = llvm.mlir.zero : !llvm.ptr
      %zero_i64 = hw.constant 0 : i64
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %lhs0 = llvm.insertvalue %null, %undef[0] : !llvm.struct<(ptr, i64)>
      %lhs = llvm.insertvalue %zero_i64, %lhs0[1] : !llvm.struct<(ptr, i64)>
      %rhs0 = llvm.insertvalue %null, %undef[0] : !llvm.struct<(ptr, i64)>
      %rhs = llvm.insertvalue %zero_i64, %rhs0[1] : !llvm.struct<(ptr, i64)>

      scf.for %idx = %zero to %one step %one {
        %match =
            func.call @"uvm_pkg::uvm_is_match"(%lhs, %rhs) : (!llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>) -> i1
        scf.if %match {
          sim.proc.print %fmt
        }
      }
      llhd.halt
    }

    hw.output
  }
}

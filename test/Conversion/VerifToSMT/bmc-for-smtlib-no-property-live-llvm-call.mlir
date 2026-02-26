// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Propertyless BMC ops are lowered to "true". Even if the circuit region
// carries unsupported LLVM ops (e.g. runtime helper calls), this should not
// fail SMT-LIB export because there is no property to prove.

func.func @for_smtlib_no_property_live_llvm_call() -> (i1) {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values []
  init {
    %c0_i1 = hw.constant 0 : i1
    %clk = seq.to_clock %c0_i1
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    verif.yield %clk : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock):
    %c0_i64 = hw.constant 0 : i64
    %ptr = llvm.call @malloc(%c0_i64) : (i64) -> !llvm.ptr
    verif.yield %ptr : !llvm.ptr
  }
  func.return %bmc : i1
}

llvm.func @malloc(i64) -> !llvm.ptr

// CHECK-LABEL: func.func @for_smtlib_no_property_live_llvm_call() -> i1
// CHECK-NEXT: [[TRUE:%.+]] = arith.constant true
// CHECK-NEXT: return [[TRUE]] : i1

// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --verify-diagnostics

llvm.func @malloc(i64) -> !llvm.ptr

func.func @for_smtlib_rejects_llvm_ops_in_bmc() -> (i1) {
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
    // expected-error @below {{for-smtlib-export does not support LLVM dialect operations inside verif.bmc regions; found 'llvm.mlir.constant'}}
    %zero = llvm.mlir.constant(0 : i64) : i64
    %ptr = llvm.call @malloc(%zero) : (i64) -> !llvm.ptr
    %cond = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to i1
    verif.assert %cond : i1
    verif.yield %cond : i1
  }
  func.return %bmc : i1
}

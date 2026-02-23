// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --split-input-file --verify-diagnostics

func.func @for_smtlib_rejects_overflow_flagged_llvm_add() -> (i1) {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values []
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
    ^bb0(%clk: !seq.clock, %x: i8):
      %one = llvm.mlir.constant(1 : i8) : i8
      // expected-error @below {{for-smtlib-export does not support LLVM dialect operations inside verif.bmc regions; found 'llvm.add'}}
      %sum = "llvm.add"(%x, %one) <{overflowFlags = 2 : i32}> : (i8, i8) -> i8
      %ok = llvm.icmp "eq" %sum, %sum : i8
      verif.assert %ok : i1
      verif.yield %x : i8
  }
  func.return %bmc : i1
}

// -----

func.func @for_smtlib_rejects_exact_llvm_udiv() -> (i1) {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values []
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
    ^bb0(%clk: !seq.clock, %x: i8):
      %two = llvm.mlir.constant(2 : i8) : i8
      // expected-error @below {{for-smtlib-export does not support LLVM dialect operations inside verif.bmc regions; found 'llvm.udiv'}}
      %div = "llvm.udiv"(%x, %two) <{isExact}> : (i8, i8) -> i8
      %ok = llvm.icmp "eq" %div, %div : i8
      verif.assert %ok : i1
      verif.yield %x : i8
  }
  func.return %bmc : i1
}

// RUN: not circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect 2>&1 | FileCheck %s

// CHECK: unsupported sequence lowering for block argument

func.func @bmc_concat_unknown_bounds() -> i1 {
  %bmc = verif.bmc bound 4 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%seq: !ltl.sequence, %sig: i1):
    %concat = ltl.concat %seq, %sig : !ltl.sequence, i1
    verif.assert %concat : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}

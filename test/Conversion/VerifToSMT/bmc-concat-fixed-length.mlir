// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s
// Test that fixed-length concatenation lowers via the BMC sequence NFA path.

// CHECK-LABEL: func.func @bmc_concat_fixed_length
// CHECK: scf.for
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> ({{.*}}!smt.bool)
func.func @bmc_concat_fixed_length() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    %rep_a = ltl.repeat %a, 2, 0 : i1
    %concat = ltl.concat %rep_a, %b : !ltl.sequence, i1
    verif.assert %concat : !ltl.sequence
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s
// Test that concat with a variable-length prefix lowers via the BMC sequence
// NFA path.

// CHECK-LABEL: func.func @bmc_concat_variable_prefix
// CHECK: scf.for
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> ({{.*}}!smt.bool)
func.func @bmc_concat_variable_prefix() -> i1 {
  %bmc = verif.bmc bound 6 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    %range_a = ltl.delay %a, 1, 1 : i1
    %concat = ltl.concat %range_a, %b : !ltl.sequence, i1
    verif.assert %concat : !ltl.sequence
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

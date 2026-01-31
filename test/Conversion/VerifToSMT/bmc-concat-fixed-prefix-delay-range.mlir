// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s
// Test that concatenation with a fixed-length prefix and a variable-length
// suffix lowers via the BMC sequence NFA path.

// CHECK-LABEL: func.func @bmc_concat_fixed_prefix_delay_range
// CHECK: scf.for
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> ({{.*}}!smt.bool)
func.func @bmc_concat_fixed_prefix_delay_range() -> i1 {
  %bmc = verif.bmc bound 6 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    %rep_a = ltl.repeat %a, 2, 0 : i1
    %range_b = ltl.delay %b, 1, 2 : i1
    %concat = ltl.concat %rep_a, %range_b : !ltl.sequence, !ltl.sequence
    verif.assert %concat : !ltl.sequence
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

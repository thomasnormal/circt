// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test that large concatenations lower via the BMC sequence NFA path without
// hitting expansion limits.
// CHECK-LABEL: func.func @bmc_concat_expansion_too_large
// CHECK: scf.for
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> ({{.*}}!smt.bool)

func.func @bmc_concat_expansion_too_large() -> i1 {
  %bmc = verif.bmc bound 4 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1, %c: i1, %d: i1):
    %p0 = ltl.delay %a, 0, 8 : i1
    %p1 = ltl.delay %b, 0, 8 : i1
    %p2 = ltl.delay %c, 0, 8 : i1
    %concat = ltl.concat %p0, %p1, %p2, %d : !ltl.sequence, !ltl.sequence, !ltl.sequence, i1
    verif.assert %concat : !ltl.sequence
    verif.yield %a, %b, %c, %d : i1, i1, i1, i1
  }
  func.return %bmc : i1
}

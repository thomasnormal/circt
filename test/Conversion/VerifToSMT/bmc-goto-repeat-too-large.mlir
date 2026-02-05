// RUN: circt-opt %s --convert-verif-to-smt -allow-unregistered-dialect | FileCheck %s

// This used to fail due to combinatorial expansion of goto/non-consecutive
// repeat. It is now handled via the NFA-based BMC lowering.

func.func @test_goto_repeat_too_large() -> i1 {
// CHECK-LABEL: func.func @test_goto_repeat_too_large
// CHECK:         scf.for
// CHECK:         func.call @bmc_circuit
// CHECK:         func.func @bmc_circuit(
// CHECK-SAME:      -> ({{.*}}, !smt.bool)
  %bmc = verif.bmc bound 18 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1):
    // base=9, more=0 => choose 8 from 17 offsets (exceeds limit)
    %seq = ltl.goto_repeat %a, 9, 0 : i1
    verif.assert %seq : !ltl.sequence
    verif.yield %a : i1
  }
  func.return %bmc : i1
}

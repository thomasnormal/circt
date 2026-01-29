// RUN: not circt-opt %s --convert-verif-to-smt -allow-unregistered-dialect 2>&1 | FileCheck %s

// CHECK: goto/non-consecutive repeat expansion too large

func.func @test_goto_repeat_too_large() -> i1 {
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

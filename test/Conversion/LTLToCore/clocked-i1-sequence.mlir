// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  hw.module @test(in %clk : i1, in %a : i1) {
    %clocked = ltl.clock %a, posedge %clk : i1
    verif.assert %clocked : !ltl.sequence
    hw.output
  }

  hw.module @test_negedge(in %clk : i1, in %a : i1) {
    %clocked = ltl.clock %a, negedge %clk : i1
    verif.assert %clocked : !ltl.sequence
    hw.output
  }

  hw.module @test_nested(in %clk : !seq.clock, in %alt : i1, in %a : i1) {
    %clocked = ltl.clock %a, negedge %alt : i1
    verif.assert %clocked : !ltl.sequence
    hw.output
  }
}

// CHECK: hw.module @test
// CHECK-DAG: seq.to_clock
// CHECK-DAG: seq.compreg sym @ltl_state
// CHECK: verif.assert
// CHECK: hw.module @test_negedge
// CHECK: comb.xor
// CHECK: seq.to_clock
// CHECK: seq.compreg sym @ltl_state
// CHECK: verif.assert
// CHECK: hw.module @test_nested
// CHECK-NOT: seq.from_clock %clk
// CHECK: comb.xor
// CHECK: seq.to_clock
// CHECK: seq.compreg sym @ltl_state
// CHECK: verif.assert

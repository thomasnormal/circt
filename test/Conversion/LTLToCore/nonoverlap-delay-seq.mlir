// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  hw.module @test(in %clock : !seq.clock, in %a : i1, in %b : i1, in %c : i1) {
    %clk = seq.from_clock %clock
    %b0 = ltl.delay %b, 0, 0 : i1
    %c0 = ltl.delay %c, 0, 0 : i1
    %seq = ltl.concat %b0, %c0 : !ltl.sequence, !ltl.sequence
    %del = ltl.delay %seq, 1, 0 : !ltl.sequence
    %prop = ltl.implication %a, %del : i1, !ltl.sequence
    verif.clocked_assert %prop, posedge %clk : !ltl.property
    hw.output
  }
}

// CHECK: hw.module @test
// CHECK: ltl_state
// CHECK-NOT: ltl_past

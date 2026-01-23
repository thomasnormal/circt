// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  hw.module @test(in %clock : !seq.clock, in %a : i1, in %b : i1) {
    %clk = seq.from_clock %clock
    %a0 = ltl.delay %a, 0, 0 : i1
    %b0 = ltl.delay %b, 3, 0 : i1
    %seq = ltl.concat %a0, %b0 : !ltl.sequence, !ltl.sequence
    verif.clocked_assert %seq, posedge %clk : !ltl.sequence
    hw.output
  }
}

// CHECK: hw.module @test
// CHECK-COUNT-4: seq.compreg sym @ltl_past

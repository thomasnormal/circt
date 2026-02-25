// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

module SVAStrongSequenceNexttimeAlways(input logic clk, a, b);
  sequence s;
    @(posedge clk) a ##1 b;
  endsequence

  // Weak nexttime over a sequence: plain delay.
  // CHECK: %[[NXT:.*]] = ltl.delay %{{.*}}, 1, 0 : !ltl.sequence
  // CHECK: verif.assert %[[NXT]] : !ltl.sequence
  assert property (nexttime s);

  // Strong nexttime over a sequence: require finite progress.
  // CHECK: %[[SNXT_EV:.*]] = ltl.eventually %[[NXT]] : !ltl.sequence
  // CHECK: %[[SNXT_STRONG:.*]] = ltl.and %[[NXT]], %[[SNXT_EV]] : !ltl.sequence, !ltl.property
  // CHECK: verif.assert %[[SNXT_STRONG]] : !ltl.property
  assert property (s_nexttime s);

  // Weak always over a sequence range: conjunction of shifted implications.
  // CHECK: %[[D1:.*]] = ltl.delay %true, 1, 0 : i1
  // CHECK: %[[P1:.*]] = ltl.implication %[[D1]], %{{.*}} : !ltl.sequence, !ltl.sequence
  // CHECK: %[[D2:.*]] = ltl.delay %true, 2, 0 : i1
  // CHECK: %[[P2:.*]] = ltl.implication %[[D2]], %{{.*}} : !ltl.sequence, !ltl.sequence
  // CHECK: %[[ALW:.*]] = ltl.and %[[P1]], %[[P2]] : !ltl.property, !ltl.property
  // CHECK: verif.assert %[[ALW]] : !ltl.property
  assert property (always [1:2] s);

  // Strong always over a sequence range: also require progress to each bound.
  // CHECK: %[[S1:.*]] = ltl.and %[[D1]], %[[P1]] : !ltl.sequence, !ltl.property
  // CHECK: %[[S2:.*]] = ltl.and %[[D2]], %[[P2]] : !ltl.sequence, !ltl.property
  // CHECK: %[[SALW_STRONG:.*]] = ltl.and %[[S1]], %[[S2]] : !ltl.property, !ltl.property
  // CHECK: verif.assert %[[SALW_STRONG]] : !ltl.property
  assert property (s_always [1:2] s);
endmodule

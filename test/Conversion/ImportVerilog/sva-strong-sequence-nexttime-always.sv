// RUN: circt-translate --import-verilog %s | FileCheck %s

module SVAStrongSequenceNexttimeAlways(input logic clk, a, b);
  sequence s;
    @(posedge clk) a ##1 b;
  endsequence

  // Weak nexttime over a sequence: plain delay.
  // CHECK: %[[NXT:.*]] = ltl.delay %{{.*}}, 1, 0 : !ltl.sequence
  // CHECK: verif.assert %[[NXT]] : !ltl.sequence
  assert property (nexttime s);

  // Strong nexttime over a sequence: require finite progress.
  // CHECK: %[[SNXT:.*]] = ltl.delay %{{.*}}, 1, 0 : !ltl.sequence
  // CHECK: %[[SNXT_EV:.*]] = ltl.eventually %[[SNXT]] : !ltl.sequence
  // CHECK: %[[SNXT_STRONG:.*]] = ltl.and %[[SNXT]], %[[SNXT_EV]] : !ltl.sequence, !ltl.property
  // CHECK: verif.assert %[[SNXT_STRONG]] : !ltl.property
  assert property (s_nexttime s);

  // Weak always over a sequence range: plain repeat.
  // CHECK: %[[ALW:.*]] = ltl.repeat %{{.*}}, 1, 1 : !ltl.sequence
  // CHECK: verif.assert %[[ALW]] : !ltl.sequence
  assert property (always [1:2] s);

  // Strong always over a sequence range: require finite progress.
  // CHECK: %[[SALW:.*]] = ltl.repeat %{{.*}}, 1, 1 : !ltl.sequence
  // CHECK: %[[SALW_EV:.*]] = ltl.eventually %[[SALW]] : !ltl.sequence
  // CHECK: %[[SALW_STRONG:.*]] = ltl.and %[[SALW]], %[[SALW_EV]] : !ltl.sequence, !ltl.property
  // CHECK: verif.assert %[[SALW_STRONG]] : !ltl.property
  assert property (s_always [1:2] s);
endmodule

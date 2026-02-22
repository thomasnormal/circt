// RUN: circt-translate --import-verilog %s | FileCheck %s

module SVABoundedEventuallySequence(input logic clk, a, b);
  sequence s;
    @(posedge clk) a ##1 b;
  endsequence

  // Weak bounded eventually on sequence: plain delay range.
  // CHECK: %[[W:.*]] = ltl.delay %{{.*}}, 1, 1 : !ltl.sequence
  // CHECK: verif.assert %[[W]] : !ltl.sequence
  assert property (eventually [1:2] s);

  // Strong bounded eventually on sequence: delay range + finite-progress
  // obligation.
  // CHECK: %[[S:.*]] = ltl.delay %{{.*}}, 2, 1 : !ltl.sequence
  // CHECK: %[[SEV:.*]] = ltl.eventually %[[S]] : !ltl.sequence
  // CHECK: %[[SS:.*]] = ltl.and %[[S]], %[[SEV]] : !ltl.sequence, !ltl.property
  // CHECK: verif.assert %[[SS]] : !ltl.property
  assert property (s_eventually [2:3] s);
endmodule

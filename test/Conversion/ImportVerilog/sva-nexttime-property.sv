// RUN: circt-translate --import-verilog %s | FileCheck %s

module SVANexttimeProperty(input logic clk, a, b, c, d);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // CHECK: %[[TRUE1:.*]] = hw.constant true
  // CHECK: %[[D1:.*]] = ltl.delay %[[TRUE1]], 1, 0 : i1
  // CHECK: %[[P1:.*]] = ltl.implication %[[D1]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: verif.assert %[[P1]] : !ltl.property
  assert property (nexttime p);

  // CHECK: %[[TRUE2:.*]] = hw.constant true
  // CHECK: %[[D2:.*]] = ltl.delay %[[TRUE2]], 2, 0 : i1
  // CHECK: %[[P2:.*]] = ltl.implication %[[D2]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: verif.assert %[[P2]] : !ltl.property
  assert property (nexttime [2] p);

  // CHECK: %[[TRUE3:.*]] = hw.constant true
  // CHECK: %[[D4:.*]] = ltl.delay %[[TRUE3]], 4, 0 : i1
  // CHECK: %[[P4:.*]] = ltl.implication %[[D4]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: verif.assert %[[P4]] : !ltl.property
  assert property (s_nexttime [4] p);

  assert property (@(posedge clk) c |-> d);
endmodule

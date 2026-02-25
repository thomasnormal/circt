// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

module SVANexttimeProperty(input logic clk, a, b, c, d);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // CHECK: %[[TRUE:.*]] = hw.constant true
  // CHECK: %[[D1:.*]] = ltl.delay %[[TRUE]], 1, 0 : i1
  // CHECK: %[[P1:.*]] = ltl.implication %[[D1]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: verif.assert %[[P1]] : !ltl.property
  assert property (nexttime p);

  // CHECK: %[[D2:.*]] = ltl.delay %[[TRUE]], 2, 0 : i1
  // CHECK: %[[P2:.*]] = ltl.implication %[[D2]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: verif.assert %[[P2]] : !ltl.property
  assert property (nexttime [2] p);

  // CHECK: %[[D4:.*]] = ltl.delay %[[TRUE]], 4, 0 : i1
  // CHECK: %[[P4:.*]] = ltl.implication %[[D4]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: %[[SP4:.*]] = ltl.and %[[D4]], %[[P4]] : !ltl.sequence, !ltl.property
  // CHECK: verif.assert %[[SP4]] : !ltl.property
  assert property (s_nexttime [4] p);

  assert property (@(posedge clk) c |-> d);
endmodule

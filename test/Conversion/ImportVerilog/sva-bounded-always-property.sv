// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s

module SVABoundedAlwaysProperty(input logic clk, a, b, c, d);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // CHECK: %[[TRUE1:.*]] = hw.constant true
  // CHECK: %[[D1:.*]] = ltl.delay %[[TRUE1]], 1, 0 : i1
  // CHECK: %[[P1:.*]] = ltl.implication %[[D1]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: %[[TRUE2:.*]] = hw.constant true
  // CHECK: %[[D2:.*]] = ltl.delay %[[TRUE2]], 2, 0 : i1
  // CHECK: %[[P2:.*]] = ltl.implication %[[D2]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: %[[AND1:.*]] = ltl.and %{{.*}}, %[[P1]], %[[P2]]
  // CHECK: verif.assert %[[AND1]] : !ltl.property
  assert property (always [0:2] p);

  // CHECK: %[[TRUE3:.*]] = hw.constant true
  // CHECK: %[[D3:.*]] = ltl.delay %[[TRUE3]], 1, 0 : i1
  // CHECK: %[[PA:.*]] = ltl.implication %[[D3]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: %[[SPA:.*]] = ltl.and %[[D3]], %[[PA]] : !ltl.sequence, !ltl.property
  // CHECK: %[[TRUE4:.*]] = hw.constant true
  // CHECK: %[[D4:.*]] = ltl.delay %[[TRUE4]], 2, 0 : i1
  // CHECK: %[[PB:.*]] = ltl.implication %[[D4]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: %[[SPB:.*]] = ltl.and %[[D4]], %[[PB]] : !ltl.sequence, !ltl.property
  // CHECK: %[[AND2:.*]] = ltl.and %[[SPA]], %[[SPB]] : !ltl.property, !ltl.property
  // CHECK: verif.assert %[[AND2]] : !ltl.property
  assert property (s_always [1:2] p);

  assert property (@(posedge clk) c |-> d);
endmodule

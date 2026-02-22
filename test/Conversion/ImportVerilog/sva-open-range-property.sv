// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s

module SVAOpenRangeProperty(input logic clk, a, b, c, d);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // CHECK: %[[T1:.*]] = hw.constant true
  // CHECK: %[[D1:.*]] = ltl.delay %[[T1]], 1, 0 : i1
  // CHECK: %[[SP:.*]] = ltl.implication %[[D1]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: %[[SSP:.*]] = ltl.and %[[D1]], %[[SP]] : !ltl.sequence, !ltl.property
  // CHECK: %[[SEV:.*]] = ltl.eventually %[[SSP]] : !ltl.property
  // CHECK: verif.assert %[[SEV]] : !ltl.property
  assert property (s_eventually [1:$] p);

  // CHECK: %[[T2:.*]] = hw.constant true
  // CHECK: %[[D2:.*]] = ltl.delay %[[T2]], 1, 0 : i1
  // CHECK: %[[AP:.*]] = ltl.implication %[[D2]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: %[[N1:.*]] = ltl.not %[[AP]] : !ltl.property
  // CHECK: %[[EV1:.*]] = ltl.eventually %[[N1]] {ltl.weak} : !ltl.property
  // CHECK: %[[ALW:.*]] = ltl.not %[[EV1]] : !ltl.property
  // CHECK: verif.assert %[[ALW]] : !ltl.property
  assert property (always [1:$] p);

  assert property (@(posedge clk) c |-> d);
endmodule

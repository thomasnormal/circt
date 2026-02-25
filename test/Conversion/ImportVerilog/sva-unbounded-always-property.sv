// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s

module SVAUnboundedAlwaysProperty(input logic clk, a, b, c, d);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // CHECK: %[[N1:.*]] = ltl.not %{{.*}} : !ltl.property
  // CHECK: %[[E1:.*]] = ltl.eventually %[[N1]] {ltl.weak} : !ltl.property
  // CHECK: %[[A1:.*]] = ltl.not %[[E1]] : !ltl.property
  // CHECK: verif.assert %[[A1]] : !ltl.property
  assert property (always p);

  // CHECK: %[[N2:.*]] = ltl.not %{{.*}} : i1
  // CHECK: %[[E2:.*]] = ltl.eventually %[[N2]] {ltl.weak} : !ltl.property
  // CHECK: %[[A2:.*]] = ltl.not %[[E2]] : !ltl.property
  // CHECK: verif.clocked_assert %[[A2]], posedge %{{.*}} : !ltl.property
  assert property (@(posedge clk) always a);

  assert property (@(posedge clk) c |-> d);
endmodule

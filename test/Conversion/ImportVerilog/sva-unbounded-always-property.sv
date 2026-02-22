// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s

module SVAUnboundedAlwaysProperty(input logic clk, a, b, c, d);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // CHECK: %[[N1:.*]] = ltl.not %{{.*}} : !ltl.property
  // CHECK: %[[E1:.*]] = ltl.eventually %[[N1]] : !ltl.property
  // CHECK: %[[A1:.*]] = ltl.not %[[E1]] : !ltl.property
  // CHECK: verif.assert %[[A1]] : !ltl.property
  assert property (always p);

  assert property (@(posedge clk) c |-> d);
endmodule

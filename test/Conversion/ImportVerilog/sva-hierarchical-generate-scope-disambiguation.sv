// RUN: circt-verilog --ir-moore --top=top %s | FileCheck %s
// REQUIRES: slang

// Hierarchical references through named generate scopes must preserve the
// generate segment. Otherwise sibling generate scopes with the same instance
// names collapse to one threaded hierarchical signal.

module leaf (
  input logic clk,
  output logic unused_err_o
);
  assign unused_err_o = clk;
endmodule

module mid (
  input logic clk
);
  if (1) begin : g1
    leaf u (.clk(clk), .unused_err_o());
  end
  if (1) begin : g2
    leaf u (.clk(clk), .unused_err_o());
  end
endmodule

module top (
  input logic clk
);
  mid m (.clk(clk));

  assert property (@(posedge clk) !m.g1.u.unused_err_o);
  assert property (@(posedge clk) !m.g2.u.unused_err_o);
endmodule

// CHECK-LABEL: moore.module private @mid
// CHECK-SAME: out g1.u.unused_err_o : !moore.ref<l1>
// CHECK-SAME: out g2.u.unused_err_o : !moore.ref<l1>
// CHECK-LABEL: moore.module @top
// CHECK: %m.g1.u.unused_err_o, %m.g2.u.unused_err_o = moore.instance "m" @mid
// CHECK: moore.read %m.g1.u.unused_err_o
// CHECK: moore.read %m.g2.u.unused_err_o

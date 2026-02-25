// RUN: circt-verilog --ir-moore --top=top %s | FileCheck %s

module prim_count (
  input logic clk,
  output logic err_o
);
  logic err_q;
  always_ff @(posedge clk)
    err_q <= ~err_q;
  assign err_o = err_q;
endmodule

module accu (
  input logic clk
);
  prim_count u_prim_count (.clk(clk), .err_o());
endmodule

module top (
  input logic clk
);
  for (genvar k = 0; k < 2; k++) begin : gen_accu
    accu u_accu (.clk(clk));
    assert property (@(posedge clk) $rose(u_accu.u_prim_count.err_o) |-> 1'b1);
  end
endmodule

// CHECK-LABEL: moore.module @top
// CHECK: moore.instance "gen_accu_0.u_accu" @accu
// CHECK: verif.clocked_assert
// CHECK: moore.instance "gen_accu_1.u_accu" @accu
// CHECK: verif.clocked_assert

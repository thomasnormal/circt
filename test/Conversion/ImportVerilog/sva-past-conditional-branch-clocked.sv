// RUN: circt-verilog --no-uvm-auto-include %s | FileCheck %s

module top(
    input logic clk,
    input logic rst_n,
    input logic [3:0] test_expr
);
  property p_increment_delta;
    @(posedge clk)
      disable iff (!rst_n)
      (!$stable(test_expr)) |-> ((test_expr > $past(test_expr)) ?
                                     (test_expr - $past(test_expr) == 4'd1) :
                                     (test_expr + (4'hf - $past(test_expr)) +
                                          4'd1 == 4'd1));
  endproperty

  a_increment_delta: assert property (p_increment_delta);
endmodule

// CHECK: hw.module @top
// CHECK: verif.clocked_assert

// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s | FileCheck %s

module top(input bit clk);
  property p;
    int v = 1;
    v == 1;
  endproperty

  a_local_var_init_rvalue: assert property (@(posedge clk) p);
endmodule

// CHECK: moore.module @top
// CHECK: moore.constant 1 : i32
// CHECK: moore.eq
// CHECK: verif.clocked_assert

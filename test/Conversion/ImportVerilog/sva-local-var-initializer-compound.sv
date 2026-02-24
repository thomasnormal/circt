// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s | FileCheck %s

module top(input bit clk);
  property p;
    int v = 0;
    (1, v += 1) ##1 (v == 1);
  endproperty

  a_local_var_compound_init: assert property (@(posedge clk) p);
endmodule

// CHECK: moore.module @top
// CHECK: moore.constant 0 : i32
// CHECK: moore.variable
// CHECK: moore.add
// CHECK: moore.past
// CHECK: verif.clocked_assert

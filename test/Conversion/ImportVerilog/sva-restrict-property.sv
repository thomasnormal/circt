// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_restrict_property(input logic clk, rst, a, b);
  // CHECK-LABEL: moore.module @sva_restrict_property

  // CHECK: verif.clocked_assume {{.*}} : !ltl.property
  restrict property (@(posedge clk) a |-> b);

  // CHECK: verif.assume {{.*}} if {{.*}} : !ltl.property
  restrict property (disable iff (rst) a |-> b);

  // CHECK: verif.clocked_assume
  initial begin
    restrict property (@(posedge clk) b |-> a);
  end
endmodule

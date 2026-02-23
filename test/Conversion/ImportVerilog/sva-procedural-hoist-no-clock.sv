// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

// CHECK-LABEL: moore.module @test_proc_hoist_no_clock
// CHECK-NOT: moore.procedure always
// CHECK: verif.clocked_assert
// CHECK-SAME: if
module test_proc_hoist_no_clock(input logic clk, a, b);
  always @(*) begin
    if (a)
      assert property (@(posedge clk) b);
  end
endmodule

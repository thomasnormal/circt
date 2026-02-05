// RUN: circt-verilog %s --parse-only | FileCheck %s

// CHECK-LABEL: moore.module @test_proc_hoist_no_clock
// CHECK: moore.procedure always
// CHECK-NOT: verif.clocked_assert
// CHECK-NOT: verif.assert
// CHECK: moore.return
// CHECK: verif.clocked_assert
module test_proc_hoist_no_clock(input logic clk, a, b);
  always @(*) begin
    if (a)
      assert property (@(posedge clk) b);
  end
endmodule

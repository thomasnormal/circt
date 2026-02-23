// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

// CHECK-LABEL: moore.module @test_proc_clock
// CHECK: moore.procedure always
// CHECK: moore.return
// CHECK: verif.clocked_assert {{.*}} posedge {{.*}} : i1
module test_proc_clock(input logic clk, a);
  always @(posedge clk) begin
    assert property ($rose(a));
  end
endmodule

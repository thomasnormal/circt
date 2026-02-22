// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaProceduralExplicitClockHoistOrder(input logic clk, input logic rst,
                                            input logic en, input logic a,
                                            input logic b);
  always_comb begin
    if (en) begin
      // CHECK: arith.andi
      // CHECK: verif.clocked_assert
      assert property (disable iff (rst) @(posedge clk) a);

      // CHECK: arith.andi
      // CHECK: verif.clocked_assume
      assume property (disable iff (rst) @(posedge clk) b);
    end
  end
endmodule

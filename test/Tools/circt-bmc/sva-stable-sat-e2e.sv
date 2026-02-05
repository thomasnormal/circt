// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module=sva_stable_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_stable_sat(input logic clk);
  logic sig;
  initial sig = 1'b0;
  always @(posedge clk) begin
    sig <= ~sig;
  end
  assert property (@(posedge clk) $stable(sig));
endmodule

// CHECK: BMC_RESULT=SAT

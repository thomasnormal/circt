// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module=sva_changed_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_changed_unsat(input logic clk);
  logic sig;
  initial sig = 1'b0;
  always @(posedge clk) begin
    sig <= ~sig;
  end
  assert property (@(posedge clk) $changed(sig));
endmodule

// CHECK: BMC_RESULT=UNSAT

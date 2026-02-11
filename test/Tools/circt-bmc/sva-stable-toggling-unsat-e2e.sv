// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 6 --ignore-asserts-until=0 --module=sva_stable_toggling_unsat - | \
// RUN:   FileCheck %s
// REQUIRES: slang
// REQUIRES: z3

module sva_stable_toggling_unsat(input logic clk);
  int cyc = 0;
  logic val = 0;

  always_ff @(posedge clk) begin
    cyc <= cyc + 1;
    val <= ~val;
  end

  assert property (@(posedge clk) cyc == 0 || !$stable(val));
endmodule

// CHECK: BMC_RESULT=UNSAT

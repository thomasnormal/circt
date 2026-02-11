// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 6 --ignore-asserts-until=0 --module=sva_local_var_disable_iff_midflight_abort_unsat - | \
// RUN:   FileCheck %s --check-prefix=JIT
// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 6 --ignore-asserts-until=0 --module=sva_local_var_disable_iff_midflight_abort_unsat - | \
// RUN:   FileCheck %s --check-prefix=SMTLIB
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_local_var_disable_iff_midflight_abort_unsat(input logic clk);
  logic reset;
  logic start;
  logic [3:0] in;
  logic [3:0] out;

  assign in = 4'd5;
  assign out = 4'd6;
  assign start = ~reset;

  // Constrain reset to toggle every sampled cycle, so each start pulse
  // (reset=0) is followed by reset=1 on the consequent cycle.
  assume property (@(posedge clk) !reset |-> ##1 reset);
  assume property (@(posedge clk) reset |-> ##1 !reset);

  property p;
    logic [3:0] x;
    @(posedge clk) disable iff (reset) (start, x = in) |-> ##1 (out == x + 4'd2);
  endproperty

  assert property (p);
endmodule

// JIT: BMC_RESULT=UNSAT
// SMTLIB: BMC_RESULT=UNSAT

// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 6 --ignore-asserts-until=0 --module=sva_local_var_disable_iff_abort_unsat - | \
// RUN:   FileCheck %s --check-prefix=JIT
// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 6 --ignore-asserts-until=0 --module=sva_local_var_disable_iff_abort_unsat - | \
// RUN:   FileCheck %s --check-prefix=SMTLIB
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_local_var_disable_iff_abort_unsat;
  logic clk = 1'b0;
  always #1 clk = ~clk;

  logic [3:0] in = 4'd5;
  logic [3:0] out = 4'd0;
  logic start = 1'b1;
  logic reset = 1'b0;

  // Start one matching attempt, then disable before the consequent cycle.
  always_ff @(posedge clk) begin
    start <= 1'b0;
    reset <= 1'b1;
  end

  property p;
    logic [3:0] x;
    @(posedge clk) disable iff (reset) (start, x = in) |-> ##1 (out == x + 4'd1);
  endproperty

  assert property (p);
endmodule

// JIT: BMC_RESULT=UNSAT
// SMTLIB: BMC_RESULT=UNSAT

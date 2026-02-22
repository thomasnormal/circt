// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 5 --ignore-asserts-until=0 --module=sva_local_var_disable_iff_parity - | \
// RUN:   FileCheck %s --check-prefix=JIT
// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 5 --ignore-asserts-until=0 --module=sva_local_var_disable_iff_parity - | \
// RUN:   FileCheck %s --check-prefix=SMTLIB
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_local_var_disable_iff_parity(
    input logic clk,
    input logic valid,
    input logic [7:0] in,
    output logic [7:0] out);
  logic reset;
  assign reset = 1'b0;

  always_ff @(posedge clk) begin
    if (valid)
      out <= in + 8'd1;
  end

  property p;
    logic [7:0] x;
    @(posedge clk) disable iff (reset) (valid, x = in) |-> ##1 (out == x + 8'd2);
  endproperty

  assert property (p);
endmodule

// JIT: BMC_RESULT=SAT
// SMTLIB: BMC_RESULT=SAT

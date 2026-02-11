// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 4 --ignore-asserts-until=0 --module=sva_sequence_subroutine_call_sat - | \
// RUN:   FileCheck %s --check-prefix=JIT
// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 4 --ignore-asserts-until=0 --module=sva_sequence_subroutine_call_sat - | \
// RUN:   FileCheck %s --check-prefix=SMTLIB
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_sequence_subroutine_call_sat(input logic clk);
  logic start;
  logic [3:0] in;
  logic [3:0] out;

  assign start = 1'b1;
  assign in = 4'd5;
  assign out = in + 4'd1;

  sequence seq;
    logic [3:0] x;
    @(posedge clk) (start, x = in, $display("seq subroutine")) ##1 (out == x + 4'd2);
  endsequence

  assert property (seq);
endmodule

// JIT: BMC_RESULT=SAT
// SMTLIB: BMC_RESULT=SAT

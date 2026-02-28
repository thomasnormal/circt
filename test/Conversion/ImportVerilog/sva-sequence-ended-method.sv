// RUN: not circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=ERR
// REQUIRES: slang

module SVASequenceEndedMethod(input logic clk, a, b);
  sequence s;
    @(posedge clk) a ##1 b;
  endsequence

  // Slang currently rejects sequence `.ended` access.
  assert property (@(posedge clk) s.ended);
endmodule

// ERR: invalid member access for type 'sequence'

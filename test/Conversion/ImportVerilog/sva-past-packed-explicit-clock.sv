// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaPastPackedExplicitClock(input logic clk);
  typedef struct packed {
    logic [1:0] f;
  } pkt_t;

  pkt_t s;

  // Packed operands in explicit-clocked $past should lower by sampling the
  // simple-bit-vector form and converting back to packed type at the result.
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: moore.detect_event posedge
  // CHECK: moore.packed_to_sbv
  // CHECK: moore.sbv_to_packed
  // CHECK: verif.assert
  assert property ($past(s, 1, @(posedge clk)) == s);
endmodule

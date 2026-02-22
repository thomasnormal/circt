// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledPacked(input logic clk);
  typedef struct packed {
    logic [1:0] f;
  } pkt_t;

  pkt_t s;

  // Packed sampled values should lower in regular assertion-clocked usage.
  // CHECK: moore.packed_to_sbv
  // CHECK: moore.past
  // CHECK: moore.eq
  // CHECK: moore.not
  // CHECK: verif.assert
  assert property (@(posedge clk) $changed(s));

  // CHECK: moore.packed_to_sbv
  // CHECK: moore.past
  // CHECK: moore.eq
  // CHECK: verif.assert
  assert property (@(posedge clk) $stable(s));
endmodule

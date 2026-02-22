// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledPackedExplicitClock(input logic clk);
  typedef struct packed {
    logic [1:0] f;
  } pkt_t;

  pkt_t s;

  // Packed sampled values with explicit clocking should lower via sampled-value
  // helper procs instead of erroring on non-IntType operands.
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: moore.detect_event posedge
  // CHECK: verif.assert
  assert property ($changed(s, @(posedge clk)));
endmodule

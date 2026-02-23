// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | circt-opt \
// RUN:   --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:   --strip-llhd-processes \
// RUN:   --lower-to-bmc="top-module=sva_seq_match_item_display bound=3" \
// RUN:   | FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

// Sequence match-item display side effects should not break formal lowering.

module sva_seq_match_item_display(input logic clk, input logic a, input logic b);
  sequence seq;
    @(posedge clk) a ##1 (b, $display("seq"));
  endsequence

  assert property (seq);
endmodule

// CHECK-BMC: verif.bmc bound {{[0-9]+}}
// CHECK-BMC: loop

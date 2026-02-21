// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | circt-opt \
// RUN:   --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:   --lower-to-bmc="top-module=sva_abort_on_e2e bound=2" \
// RUN:   | FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

// End-to-end SV -> HW -> LTL/Core -> BMC coverage for accept_on/reject_on.

// CHECK-BMC: verif.bmc bound {{[0-9]+}}

module sva_abort_on_e2e(input logic clk, rst_n, a, b);
  assert property (@(posedge clk) accept_on(!rst_n) (a |-> b));
  assert property (@(posedge clk) reject_on(!rst_n) (a |-> b));
endmodule

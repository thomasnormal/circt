// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | circt-opt \
// RUN:   --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:   --lower-to-bmc="top-module=sva_sync_abort_on_e2e bound=2" \
// RUN:   | FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

// End-to-end SV -> HW -> LTL/Core -> BMC coverage for sync_accept_on /
// sync_reject_on.

// CHECK-BMC: verif.bmc bound {{[0-9]+}}

module sva_sync_abort_on_e2e(input logic clk, c, a, b);
  assert property (@(posedge clk) sync_accept_on(c) (a |-> b));
  assert property (@(posedge clk) sync_reject_on(c) (a |-> b));
endmodule

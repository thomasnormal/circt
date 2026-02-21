// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | circt-opt \
// RUN:   --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:   --lower-to-bmc="top-module=sva_strong_weak_e2e bound=2" \
// RUN:   | FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

// End-to-end SV -> HW -> LTL/Core -> BMC coverage for strong/weak properties.

// CHECK-BMC: verif.bmc bound {{[0-9]+}}

module sva_strong_weak_e2e(input logic clk, a, b);
  assert property (@(posedge clk) strong(a ##1 b));
  assert property (@(posedge clk) weak(a ##1 b));
endmodule

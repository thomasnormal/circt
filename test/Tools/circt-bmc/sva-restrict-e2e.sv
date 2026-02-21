// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | circt-opt \
// RUN:   --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:   --lower-to-bmc="top-module=sva_restrict bound=2" \
// RUN:   | FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

// End-to-end SV -> HW -> LTL/Core -> BMC coverage for restrict statements.

// CHECK-BMC: verif.bmc bound {{[0-9]+}}
// CHECK-BMC: verif.assume

module sva_restrict(input logic clk, input logic a, input logic b);
  restrict property (@(posedge clk) a |-> b);
endmodule

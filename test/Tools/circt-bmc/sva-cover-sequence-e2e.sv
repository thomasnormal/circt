// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | circt-opt \
// RUN:   --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:   --lower-to-bmc="top-module=sva_cover_sequence bound=2" \
// RUN:   | FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

// End-to-end SV -> HW -> LTL/Core -> BMC coverage for cover sequence.

// CHECK-BMC: verif.bmc bound {{[0-9]+}}
// CHECK-BMC: verif.assume

module sva_cover_sequence(input logic clk, input logic a, input logic b);
  cover sequence (@(posedge clk) a ##1 b);
endmodule

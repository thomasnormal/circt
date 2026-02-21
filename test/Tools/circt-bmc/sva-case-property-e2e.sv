// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | circt-opt \
// RUN:   --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:   --lower-to-bmc="top-module=sva_case_property_e2e bound=2" \
// RUN:   | FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

// End-to-end SV -> HW -> LTL/Core -> BMC coverage for case property expressions.

// CHECK-BMC: verif.bmc bound {{[0-9]+}}

module sva_case_property_e2e(input logic clk, sel, a, b, c);
  assert property (@(posedge clk)
    case (sel)
      1'b0: a;
      1'b1: b;
      default: c;
    endcase
  );
endmodule

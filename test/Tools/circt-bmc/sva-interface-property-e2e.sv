// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | circt-opt \
// RUN:   --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:   --lower-to-bmc="top-module=sva_interface_property_e2e bound=5" \
// RUN:   | FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang
// XFAIL: *
// Interface properties with four-state values generate LLVM loads with non-LLVM
// struct types (!hw.struct<value: i1, unknown: i1>) which fail during lowering.

// End-to-end SV -> HW -> LTL/Core -> BMC coverage for interface property clocks.

// CHECK-BMC: verif.bmc bound {{[0-9]+}}
// CHECK-BMC: loop

interface ifc(input logic clk);
  logic a;
  logic b;
  property p;
    @(posedge clk) a |-> b;
  endproperty
endinterface

module sva_interface_property_e2e(input logic clk, input logic a, input logic b);
  ifc i(clk);
  assign i.a = a;
  assign i.b = b;
  assert property (i.p);
endmodule

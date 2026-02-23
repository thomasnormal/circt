// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | circt-opt \
// RUN:   --lower-clocked-assert-like --lower-ltl-to-core \
// RUN:   --externalize-registers='allow-multi-clock=true' \
// RUN:   --lower-to-bmc="top-module=sva_multiclock bound=5 allow-multi-clock" \
// RUN:   | FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

// End-to-end SV -> HW -> LTL/Core -> BMC coverage for multi-clock SVA.

// CHECK-BMC: verif.bmc bound 20
// CHECK-BMC: loop
// CHECK-BMC: ^bb0(%{{.*}}: !seq.clock, %{{.*}}: !seq.clock, %{{.*}}: i32):

module sva_multiclock(input logic clk0, clk1, a, b);
  property p0;
    @(posedge clk0) a |-> b;
  endproperty
  property p1;
    @(posedge clk1) b |-> a;
  endproperty
  assert property (p0);
  assert property (p1);
endmodule

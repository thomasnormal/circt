// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

interface ifc(input logic clk);
  logic a;
  logic b;
  property p;
    @(posedge clk) a |-> b;
  endproperty
endinterface

// CHECK-LABEL: moore.interface @ifc
// CHECK: moore.interface.signal @clk
// CHECK: moore.interface.signal @a
// CHECK: moore.interface.signal @b

module top(input logic clk, input logic a, input logic b);
  ifc i(clk);
  assign i.a = a;
  assign i.b = b;
  assert property (i.p);

  // CHECK-LABEL: moore.module @top
  // CHECK: moore.interface.instance {{.*}}@ifc
  // CHECK: moore.virtual_interface.signal_ref %{{.*}}[@a]
  // CHECK: moore.virtual_interface.signal_ref %{{.*}}[@b]
  // CHECK: ltl.implication
  // CHECK: ltl.clock
  // CHECK: verif.assert
endmodule

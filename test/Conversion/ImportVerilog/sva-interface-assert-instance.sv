// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

interface Ifc(input logic clk, input logic d);
  assert property (@(posedge clk) $changed(d));
endinterface

module SvaInterfaceAssertInstance(input logic clk, input logic d);
  Ifc i(clk, d);

  // CHECK: moore.module @SvaInterfaceAssertInstance
  // CHECK: moore.virtual_interface.signal_ref
  // CHECK: moore.past
  // CHECK: verif.assert
endmodule

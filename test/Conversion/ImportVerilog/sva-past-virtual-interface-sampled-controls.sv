// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

interface SvaPastVifIfc;
  logic sig;
endinterface

module SvaPastVirtualInterfaceSampledControls(input logic clk, input logic a);
  virtual SvaPastVifIfc vif;

  property p_vif_past;
    @(posedge clk) $past(vif, 1, 1'b1, @(posedge clk)) == vif;
  endproperty
  assert property (p_vif_past);

  property p_vif_past_disable;
    disable iff (!a) @(posedge clk) $past(vif, 2, 1'b1, @(posedge clk)) != vif;
  endproperty
  cover property (p_vif_past_disable);
endmodule

// CHECK-LABEL: moore.module @SvaPastVirtualInterfaceSampledControls
// CHECK: moore.wait_event
// CHECK: moore.detect_event posedge
// CHECK: moore.virtual_interface_cmp
// CHECK: verif.{{(clocked_)?}}assert
// CHECK: verif.{{(clocked_)?}}cover

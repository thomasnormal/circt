// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

interface stable_iface;
  logic sig;
endinterface

class stable_class;
  int x;
endclass

module SvaStableHandleTypesSupported(input logic clk);
  stable_iface iface_inst();
  virtual stable_iface vif;
  stable_class ch;

  covergroup cg_t;
    coverpoint clk;
  endgroup
  cg_t cg;

  initial begin
    ch = new();
    vif = iface_inst;
    cg = new();
  end

  property p_stable_handles;
    @(posedge clk) $stable(ch) && $stable(vif) && $changed(cg);
  endproperty
  assert property (p_stable_handles);
endmodule

// CHECK-LABEL: moore.module @SvaStableHandleTypesSupported
// CHECK: moore.class_handle_cmp
// CHECK: moore.virtual_interface_cmp
// CHECK: moore.covergroup_handle_cmp
// CHECK: verif.{{(clocked_)?}}assert

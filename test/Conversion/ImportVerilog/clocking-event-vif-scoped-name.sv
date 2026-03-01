// RUN: circt-verilog --no-uvm-auto-include %s --ir-moore 2>&1 | FileCheck %s
// CHECK-NOT: error:
// CHECK-LABEL: func.func private @"drv::run"
// CHECK: moore.wait_event {
// CHECK: moore.virtual_interface.signal_ref %{{.*}}[@aclk]
// CHECK: moore.detect_event posedge

interface ifc(input aclk);
  logic data;
  clocking cb @(posedge aclk);
    input data;
  endclocking
endinterface

class drv;
  virtual ifc vif;

  task run();
    @(vif.cb);
  endtask
endclass

module top;
  logic clk;
  ifc if0(clk);
  drv d;

  initial begin
    d = new();
    d.vif = if0;
    d.run();
  end
endmodule

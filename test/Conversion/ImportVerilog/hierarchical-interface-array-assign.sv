// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test hierarchical interface array element assignment to virtual interface.
// CHECK-LABEL: moore.module @Top()
// CHECK: moore.instance "m" @M
// CHECK: moore.blocking_assign %vif, %{{.*}} : virtual_interface<@IF>
// CHECK: func.call @"IF::ping{{(_[0-9]+)?}}"

interface IF(input bit clk);
  task ping();
    @(posedge clk);
  endtask
endinterface

module M(input bit clk);
  IF ifs[2](clk);
endmodule

module Top;
  bit clk = 0;
  M m(clk);
  virtual IF vif;

  initial begin
    vif = m.ifs[1];
    vif.ping();
  end
endmodule

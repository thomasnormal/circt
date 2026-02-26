// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test virtual interface binding through module-instance arrays:
//   vif = module_array[idx].interface_array[idx];
// CHECK-LABEL: moore.module @Top()
// CHECK: moore.blocking_assign %vif, %{{.*}} : virtual_interface<@IF>
// CHECK: moore.blocking_assign %vif, %{{.*}} : virtual_interface<@IF>
// CHECK: func.call @"IF::ping{{(_[0-9]+)?}}"

interface IF(input bit clk);
  task ping();
    @(posedge clk);
  endtask
endinterface

module Agent(input bit clk);
  IF ifs[2](clk);
endmodule

module Top #(parameter int AIDX = 1, parameter int IIDX = 1);
  bit clk = 0;
  Agent a[2](clk);
  localparam int LIDX = 1;
  virtual IF vif;

  initial begin
    vif = a[AIDX].ifs[IIDX];
    vif = a[LIDX].ifs[1 - 0];
    vif.ping();
  end
endmodule

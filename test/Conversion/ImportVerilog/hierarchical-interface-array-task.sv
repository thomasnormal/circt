// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test hierarchical interface task calls through module.interface_array[idx].task().
// This is common in BFMs that keep channel interfaces in arrays.
// CHECK-LABEL: moore.module @Top()
// CHECK: moore.instance "agent" @Agent
// CHECK: moore.procedure initial {
// CHECK: moore.read %{{.*}} : <virtual_interface<@DriverBFM>>
// CHECK: func.call @"DriverBFM::ping{{(_[0-9]+)?}}"

interface DriverBFM(input bit clk);
  task ping();
    @(posedge clk);
  endtask
endinterface

module Agent(input bit clk);
  DriverBFM driverBFM[2](clk);
endmodule

module Top;
  bit clk = 0;
  Agent agent(clk);

  initial begin
    agent.driverBFM[1].ping();
  end
endmodule

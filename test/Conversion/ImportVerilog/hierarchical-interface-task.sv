// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test hierarchical interface task calls through module.interface.task() pattern
// This is common in UVM verification environments like AHB AVIP
// CHECK-LABEL: moore.module @TopLevel()
// CHECK: %agentBFM.driverBFM = moore.instance "agentBFM" @AgentBFM
// CHECK: moore.procedure initial {
// CHECK: moore.read %agentBFM.driverBFM : <virtual_interface<@DriverBFM>>
// CHECK: func.call @"DriverBFM::waitForResetn{{(_[0-9]+)?}}"
// CHECK: moore.read %agentBFM.driverBFM : <virtual_interface<@DriverBFM>>
// CHECK: func.call @"DriverBFM::driveToBFM"

interface DriverBFM(input bit clk, input bit resetn);
  logic [7:0] data;

  task waitForResetn();
    @(negedge resetn);
    @(posedge resetn);
  endtask

  task driveToBFM(input logic [7:0] d);
    @(posedge clk);
    data <= d;
  endtask
endinterface

module AgentBFM(input bit clk, input bit resetn);
  DriverBFM driverBFM(clk, resetn);

  initial begin
    // Direct interface task call - should work
    driverBFM.waitForResetn();
    driverBFM.driveToBFM(8'hAB);
  end
endmodule

// First test: does direct interface call work?
module DirectTest;
  bit clk = 0;
  bit resetn = 1;

  DriverBFM driverBFM(clk, resetn);

  initial begin
    // Direct interface task call in the same scope
    driverBFM.waitForResetn();
  end
endmodule

module TopLevel;
  bit clk = 0;
  bit resetn = 1;

  AgentBFM agentBFM(clk, resetn);

  initial begin
    // Hierarchical interface task call - this is the failing pattern
    // module.interface.task()
    agentBFM.driverBFM.waitForResetn();
    agentBFM.driverBFM.driveToBFM(8'hCD);
  end
endmodule

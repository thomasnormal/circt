// RUN: circt-verilog %s --ir-moore | FileCheck %s
// XFAIL: *

// Test virtual interface task calls (like UVM BFM pattern)

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

class DriverProxy;
  virtual DriverBFM vif;

  task run();
    // Virtual interface method call - this is the AHB AVIP pattern
    vif.waitForResetn();
    vif.driveToBFM(8'hAB);
  endtask
endclass

// CHECK: func.call @"DriverBFM::waitForResetn

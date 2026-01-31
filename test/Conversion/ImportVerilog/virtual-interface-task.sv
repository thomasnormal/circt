// RUN: circt-verilog %s --ir-moore | FileCheck %s

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

// CHECK-LABEL: moore.interface @DriverBFM
// CHECK:         moore.interface.signal @clk : !moore.i1
// CHECK:         moore.interface.signal @resetn : !moore.i1
// CHECK:         moore.interface.signal @data : !moore.l8

// CHECK-LABEL: func.func private @"DriverBFM::waitForResetn"
// CHECK:         moore.wait_event
// CHECK:           moore.detect_event negedge
// CHECK:         moore.wait_event
// CHECK:           moore.detect_event posedge

// CHECK-LABEL: func.func private @"DriverBFM::driveToBFM"
// CHECK:         moore.wait_event
// CHECK:           moore.detect_event posedge
// CHECK:         moore.nonblocking_assign

class DriverProxy;
  virtual DriverBFM vif;

  task run();
    // Virtual interface method call - this is the AHB AVIP pattern
    vif.waitForResetn();
    vif.driveToBFM(8'hAB);
  endtask
endclass

// CHECK-LABEL: moore.class.classdecl @DriverProxy
// CHECK:         moore.class.propertydecl @vif : !moore.virtual_interface<@DriverBFM>

// CHECK-LABEL: func.func private @"DriverProxy::run"
// CHECK:         call @"DriverBFM::waitForResetn"
// CHECK:         call @"DriverBFM::driveToBFM"

// Top-level module to trigger class elaboration
module test_top;
  DriverProxy proxy;
  virtual DriverBFM vif;

  initial begin
    proxy = new();
    proxy.vif = vif;
    proxy.run();
  end
endmodule

// CHECK-LABEL: moore.module @test_top
// CHECK:         %proxy = moore.variable : <class<@DriverProxy>>
// CHECK:         moore.class.new : <@DriverProxy>
// CHECK:         func.call @"DriverProxy::run"

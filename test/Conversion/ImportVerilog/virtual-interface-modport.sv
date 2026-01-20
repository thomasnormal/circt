// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test virtual interface with modport - UVM pattern for constrained access
// This tests modport-qualified virtual interface references with exported tasks.

// Interface with modports
// CHECK-LABEL: moore.interface @modport_if
interface modport_if(input bit clk);
  logic [7:0] data;
  logic       req;
  logic       ack;

  // Task in interface for use with modport
  task wait_ack();
    @(posedge clk);
    while (!ack) @(posedge clk);
  endtask

  // Task: send_data - master-oriented task
  // CHECK-LABEL: func.func private @"modport_if::send_data"
  // CHECK-SAME: (%{{.*}}: !moore.virtual_interface<@modport_if>, %{{.*}}: !moore.l8)
  // CHECK: moore.wait_event
  task send_data(input logic [7:0] d);
    @(posedge clk);
    data <= d;
    req <= 1'b1;
    wait_ack();
    req <= 1'b0;
  endtask

  // Modport with exported task
  modport master(
    output data,
    output req,
    input  ack,
    import task send_data(input logic [7:0] d)
  );

  modport slave(
    input  data,
    input  req,
    output ack
  );

endinterface

// Driver using modport-qualified virtual interface
// CHECK-LABEL: moore.class.classdecl @modport_driver
class modport_driver;
  // Virtual interface with modport qualifier
  // CHECK: moore.class.propertydecl @vif : !moore.virtual_interface<@modport_if::@master>
  virtual modport_if.master vif;

  // Task: drive - calls interface task through modport virtual interface
  // The key feature: modport-qualified vif is converted to base interface type
  // CHECK-LABEL: func.func private @"modport_driver::drive"
  // CHECK: moore.conversion %{{.*}} : !moore.virtual_interface<@modport_if::@master> -> !moore.virtual_interface<@modport_if>
  // CHECK: call @"modport_if::send_data"
  task drive(input logic [7:0] data);
    vif.send_data(data);
  endtask

  // Task: run
  task run();
    forever begin
      drive(8'h42);
    end
  endtask
endclass

// Top module for elaboration
// CHECK-LABEL: moore.module @test_modport
module test_modport;
  bit clk;
  modport_if dut_if(clk);

  modport_driver drv;

  initial begin
    drv = new();
    // Note: Assigning interface to modport-qualified virtual interface
    drv.vif = dut_if;
    drv.run();
  end

  always #5 clk = ~clk;
endmodule

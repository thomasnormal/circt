// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test direct access to modport signals through a virtual interface stored
// in a class property. This tests the fix for accessing signals like clk, data
// through a modport-qualified virtual interface in UVM-style patterns.

interface bus_if(input clk);
  logic [7:0] data;
  logic valid;
  logic ready;

  modport master(input clk, output data, output valid, input ready);
  modport slave(input clk, input data, input valid, output ready);
endinterface

// CHECK-LABEL: moore.class.classdecl @master_driver
class master_driver;
  // Virtual interface with modport qualifier
  // CHECK: moore.class.propertydecl @vif : !moore.virtual_interface<@bus_if::@master>
  virtual interface bus_if.master vif;

  function new(virtual interface bus_if.master vif);
    this.vif = vif;
  endfunction

  // Test: Direct signal access through modport-qualified virtual interface
  // This is the key pattern that needed to be fixed.
  // CHECK-LABEL: func.func private @"master_driver::send"
  // CHECK: moore.wait_event
  // CHECK: moore.virtual_interface.signal_ref %{{.*}}[@clk]
  // CHECK: moore.virtual_interface.signal_ref %{{.*}}[@data]
  // CHECK: moore.virtual_interface.signal_ref %{{.*}}[@valid]
  // CHECK: moore.virtual_interface.signal_ref %{{.*}}[@ready]
  task send(input logic [7:0] d);
    // Access clk through modport (input port)
    @(posedge vif.clk);
    // Write to data through modport (output port)
    vif.data <= d;
    vif.valid <= 1;
    // Wait for ready through modport (input port)
    @(posedge vif.clk);
    while (!vif.ready) @(posedge vif.clk);
    vif.valid <= 0;
  endtask
endclass

// CHECK-LABEL: moore.class.classdecl @slave_responder
class slave_responder;
  // Virtual interface with slave modport
  // CHECK: moore.class.propertydecl @vif : !moore.virtual_interface<@bus_if::@slave>
  virtual interface bus_if.slave vif;

  function new(virtual interface bus_if.slave vif);
    this.vif = vif;
  endfunction

  // CHECK-LABEL: func.func private @"slave_responder::respond"
  // CHECK: moore.wait_event
  // CHECK: moore.virtual_interface.signal_ref %{{.*}}[@clk]
  // CHECK: moore.virtual_interface.signal_ref %{{.*}}[@valid]
  // CHECK: moore.virtual_interface.signal_ref %{{.*}}[@ready]
  task respond();
    @(posedge vif.clk);
    while (!vif.valid) @(posedge vif.clk);
    repeat(2) @(posedge vif.clk);
    vif.ready <= 1;
    @(posedge vif.clk);
    vif.ready <= 0;
  endtask
endclass

// CHECK-LABEL: moore.module @test_modport_class_access
module test_modport_class_access;
  logic clk = 0;
  bus_if intf(clk);
  master_driver master;
  slave_responder slave;

  initial begin
    master = new(intf.master);
    slave = new(intf.slave);
    fork
      master.send(8'hAB);
      slave.respond();
    join
    $finish;
  end

  always #5 clk = ~clk;
endmodule

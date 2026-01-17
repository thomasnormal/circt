// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test multiple classes using the same virtual interface type.
// This test verifies that the interface is only declared once, even when
// multiple classes reference the same virtual interface type.

// Define an interface
interface my_if;
  logic [7:0] data;
  logic valid;
  logic ready;

  task send(input logic [7:0] val);
    data = val;
    valid = 1;
  endtask

  function logic [7:0] receive();
    return data;
  endfunction
endinterface

// CHECK: moore.interface @my_if
// CHECK-NOT: moore.interface @my_if_0

// First class using virtual interface
class driver;
  virtual my_if vif;

  task run();
    vif.data = 8'h42;
    vif.valid = 1;
  endtask
endclass

// CHECK-LABEL: moore.class.classdecl @driver
// CHECK:         moore.class.propertydecl @vif : !moore.virtual_interface<@my_if>

// Second class using the SAME virtual interface type
class monitor;
  virtual my_if vif;

  task run();
    if (vif.valid) begin
      vif.ready = 1;
    end
  endtask
endclass

// CHECK-LABEL: moore.class.classdecl @monitor
// CHECK:         moore.class.propertydecl @vif : !moore.virtual_interface<@my_if>

// Third class using the same virtual interface type
class scoreboard;
  virtual my_if vif;

  function logic [7:0] check_data();
    return vif.data;
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @scoreboard
// CHECK:         moore.class.propertydecl @vif : !moore.virtual_interface<@my_if>

// A class that uses the virtual interface via another class
class environment;
  driver drv;
  monitor mon;
  scoreboard sb;

  task connect_vifs(virtual my_if vif);
    drv.vif = vif;
    mon.vif = vif;
    sb.vif = vif;
  endtask
endclass

// CHECK-LABEL: moore.class.classdecl @environment

// Module to elaborate the test
module test;
  my_if intf();
  environment env;

  initial begin
    env = new();
    env.connect_vifs(intf);
  end
endmodule

// CHECK-LABEL: moore.module @test

// Additional test: ensure different interfaces are not incorrectly deduplicated
interface other_if;
  logic [15:0] addr;
  logic [31:0] data;
endinterface

// CHECK: moore.interface @other_if
// CHECK-NOT: moore.interface @other_if_0

class mixed_driver;
  virtual my_if vif1;
  virtual other_if vif2;
endclass

// CHECK-LABEL: moore.class.classdecl @mixed_driver
// CHECK:         moore.class.propertydecl @vif1 : !moore.virtual_interface<@my_if>
// CHECK:         moore.class.propertydecl @vif2 : !moore.virtual_interface<@other_if>

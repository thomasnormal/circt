// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test that inout interface ports are properly handled.

interface simple_if;
  wire data;
  wire [7:0] bus;
  // CHECK: moore.interface @simple_if
  // CHECK: moore.interface.modport @master (inout @data, inout @bus)
  modport master(inout data, inout bus);
  modport slave(inout data, inout bus);
endinterface

// CHECK: moore.module
module inout_port_test(
  simple_if.master iface
);
  // CHECK: moore.virtual_interface.signal_ref
  // CHECK: moore.assign
  assign iface.data = 1'b1;
endmodule

// Test instantiation with inout interface ports
// CHECK: moore.module @top_module
module top_module;
  simple_if my_if();

  // CHECK: moore.instance "dut"
  inout_port_test dut(.iface(my_if.master));

  wire result;
  assign result = my_if.data;
endmodule

// Test a more realistic scenario with bidirectional data
// CHECK: moore.interface @i3c_like_if
interface i3c_like_if;
  logic scl_i;
  logic scl_o;
  logic sda_i;
  logic sda_o;
  wire sda_io;  // bidirectional - must be wire for inout

  // CHECK: moore.interface.modport @controller (input @scl_i, output @scl_o, input @sda_i, output @sda_o, inout @sda_io)
  modport controller(
    input scl_i,
    output scl_o,
    input sda_i,
    output sda_o,
    inout sda_io
  );

  modport target(
    input scl_i,
    output scl_o,
    input sda_i,
    output sda_o,
    inout sda_io
  );
endinterface

// CHECK: moore.module
module i3c_controller(
  i3c_like_if.controller bus
);
  // Module can drive the inout port
  assign bus.sda_io = bus.sda_o;
endmodule

// CHECK: moore.module @i3c_top
module i3c_top;
  i3c_like_if bus_if();

  // CHECK: moore.instance "ctrl"
  i3c_controller ctrl(.bus(bus_if.controller));

  wire sda_external;
  assign sda_external = bus_if.sda_io;
endmodule

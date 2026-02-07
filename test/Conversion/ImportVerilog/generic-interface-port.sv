// RUN: circt-verilog --ir-moore %s --no-uvm-auto-include | FileCheck %s
// REQUIRES: slang

// Test generic interface ports (IEEE 1800-2017 §25.5)
// A generic interface port can be connected to any interface instance.

interface simple_if;
  logic data;
  logic valid;
endinterface

// Module with a generic interface port - resolved from connection site
module consumer(interface bus);
  logic out;
  assign out = bus.data & bus.valid;
endmodule

// CHECK-LABEL: moore.interface @simple_if
// CHECK: moore.interface.signal @data : !moore.l1
// CHECK: moore.interface.signal @valid : !moore.l1

// CHECK-LABEL: moore.module private @consumer
// CHECK-SAME: in %bus : !moore.ref<virtual_interface<@simple_if>>

// CHECK-LABEL: moore.module @top
module top;
  simple_if bus_inst();

  // CHECK: moore.instance "c" @consumer
  consumer c(.bus(bus_inst));
endmodule

// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

interface bus_if(input logic clk);
  logic data;
endinterface

module consumer(bus_if bus);
  // CHECK-LABEL: moore.module private @consumer
  // CHECK-SAME: in %bus :
  initial begin
    // CHECK: moore.procedure initial
    // CHECK: moore.read %bus : <virtual_interface<@bus_if>>
    // CHECK: moore.virtual_interface.signal_ref {{.*}}[@data] : <@bus_if> -> <l1>
    // CHECK: moore.read %bus : <virtual_interface<@bus_if>>
    // CHECK: moore.virtual_interface.signal_ref {{.*}}[@clk] : <@bus_if> -> <l1>
    // CHECK: moore.read {{.*}} : <l1>
    // CHECK: moore.blocking_assign {{.*}} : l1
    bus.data = bus.clk;
  end
endmodule

module top(input logic clk);
  bus_if bus(clk);
  consumer u_consumer(bus);
  // CHECK: %bus = moore.interface.instance @bus_if : <virtual_interface<@bus_if>>
  // CHECK: moore.instance "u_consumer"
endmodule

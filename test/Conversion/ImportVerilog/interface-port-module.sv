// RUN: circt-verilog --ir-moore %s | FileCheck %s

interface bus_if(input logic clk);
  logic data;
endinterface

module consumer(bus_if bus);
  // CHECK-LABEL: moore.module private @consumer
  // CHECK-SAME: (in %[[BUS:.*]] : !moore.ref<virtual_interface<@bus_if>>)
  initial begin
    // CHECK: moore.procedure initial
    // CHECK: %[[BUS_DATA:.*]] = moore.read %[[BUS]] : <virtual_interface<@bus_if>>
    // CHECK: %[[DATA_REF:.*]] = moore.virtual_interface.signal_ref %[[BUS_DATA]][@data] : <@bus_if> -> <l1>
    // CHECK: %[[BUS_CLK:.*]] = moore.read %[[BUS]] : <virtual_interface<@bus_if>>
    // CHECK: %[[CLK_REF:.*]] = moore.virtual_interface.signal_ref %[[BUS_CLK]][@clk] : <@bus_if> -> <l1>
    // CHECK: %[[CLK_VAL:.*]] = moore.read %[[CLK_REF]] : <l1>
    // CHECK: moore.blocking_assign %[[DATA_REF]], %[[CLK_VAL]] : l1
    bus.data = bus.clk;
  end
endmodule

module top(input logic clk);
  bus_if bus(clk);
  consumer u_consumer(bus);
  // CHECK: %bus = moore.interface.instance @bus_if : <virtual_interface<@bus_if>>
  // CHECK: moore.instance "u_consumer"
endmodule

// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test assignment of interface instance to virtual interface variable

interface simple_if;
  logic data;
  logic [7:0] value;
endinterface

// CHECK-LABEL: moore.module @test
module test;
  // CHECK: %intf = moore.interface.instance @simple_if : <virtual_interface<@simple_if>>
  simple_if intf();

  // CHECK: %vif = moore.variable : <virtual_interface<@simple_if
  virtual simple_if vif;

  initial begin
    // CHECK: [[CONV:%.*]] = moore.conversion %intf : !moore.ref<virtual_interface<@simple_if>> -> !moore.virtual_interface
    // CHECK: moore.blocking_assign %vif, [[CONV]]
    vif = intf;

    // Access through virtual interface
    // CHECK: [[VIF_READ1:%.*]] = moore.read %vif
    // CHECK: moore.virtual_interface.signal_ref [[VIF_READ1]][@data]
    vif.data = 1;

    // CHECK: [[VIF_READ2:%.*]] = moore.read %vif
    // CHECK: moore.virtual_interface.signal_ref [[VIF_READ2]][@value]
    vif.value = 8'hAB;
  end
endmodule

// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test direct interface member access (non-virtual interface)
// This tests accessing interface signals directly via the instance name
// (e.g., intf.clk) rather than through a virtual interface variable.

interface my_if;
  logic clk;
  logic [7:0] data;
endinterface

// CHECK-LABEL: moore.module @test_direct_access
module test_direct_access;
  // CHECK: %intf = moore.interface.instance @my_if : <virtual_interface<@my_if>>
  my_if intf();

  logic result;
  logic [7:0] captured_data;

  initial begin
    // Write to direct interface member
    // CHECK: [[VIF1:%.*]] = moore.read %intf : <virtual_interface<@my_if>>
    // CHECK: [[REF1:%.*]] = moore.virtual_interface.signal_ref [[VIF1]][@clk] : <@my_if> -> <l1>
    // CHECK: moore.blocking_assign [[REF1]]
    intf.clk = 1;

    // CHECK: [[VIF2:%.*]] = moore.read %intf : <virtual_interface<@my_if>>
    // CHECK: [[REF2:%.*]] = moore.virtual_interface.signal_ref [[VIF2]][@data] : <@my_if> -> <l8>
    // CHECK: moore.blocking_assign [[REF2]]
    intf.data = 8'hAB;

    // Read from direct interface member
    // CHECK: [[VIF3:%.*]] = moore.read %intf : <virtual_interface<@my_if>>
    // CHECK: [[REF3:%.*]] = moore.virtual_interface.signal_ref [[VIF3]][@clk] : <@my_if> -> <l1>
    // CHECK: [[READ3:%.*]] = moore.read [[REF3]] : <l1>
    // CHECK: moore.blocking_assign %result, [[READ3]]
    result = intf.clk;

    // CHECK: [[VIF4:%.*]] = moore.read %intf : <virtual_interface<@my_if>>
    // CHECK: [[REF4:%.*]] = moore.virtual_interface.signal_ref [[VIF4]][@data] : <@my_if> -> <l8>
    // CHECK: [[READ4:%.*]] = moore.read [[REF4]] : <l8>
    // CHECK: moore.blocking_assign %captured_data, [[READ4]]
    captured_data = intf.data;
  end
endmodule

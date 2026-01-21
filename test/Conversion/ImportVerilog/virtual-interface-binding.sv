// RUN: circt-verilog %s --ir-moore 2>&1 | FileCheck %s

// Test virtual interface runtime binding - the critical pattern for UVM testbenches
// This tests `driver.vif = apb_if` style assignments that bind a virtual interface
// to a concrete interface instance at runtime.

// Simple interface for testing
// CHECK-LABEL: moore.interface @simple_bus
interface simple_bus;
  logic [7:0] data;
  logic       valid;
  logic       ready;
endinterface

// Driver class with virtual interface
// CHECK-LABEL: moore.class.classdecl @driver
// CHECK: moore.class.propertydecl @vif : !moore.virtual_interface<@simple_bus>
class driver;
  virtual simple_bus vif;
endclass

// Module that performs virtual interface binding
// CHECK-LABEL: moore.module @test_binding
module test_binding;
  // Interface instance - should create a ref to virtual interface
  // CHECK: %bus = moore.interface.instance @simple_bus : <virtual_interface<@simple_bus>>
  simple_bus bus();

  // Driver instance
  // CHECK: %drv = moore.variable : <class<@driver>>
  driver drv;

  initial begin
    // Create driver
    // CHECK: moore.class.new
    drv = new();

    // Virtual interface binding - this is the key operation
    // The interface instance 'bus' should be converted to a virtual interface value
    // and assigned to the drv.vif property
    // CHECK: moore.class.property_ref %{{.*}}[@vif]
    // CHECK: moore.conversion %bus : !moore.ref<virtual_interface<@simple_bus>> -> !moore.virtual_interface<@simple_bus>
    // CHECK: moore.blocking_assign
    drv.vif = bus;
  end
endmodule

// Test multiple bindings and rebinding
// CHECK-LABEL: moore.module @test_multiple_bindings
module test_multiple_bindings;
  simple_bus bus1();
  simple_bus bus2();
  driver drv1, drv2;

  initial begin
    drv1 = new();
    drv2 = new();

    // Bind different interfaces to different drivers
    // CHECK: moore.conversion %bus1
    // CHECK: moore.blocking_assign
    drv1.vif = bus1;

    // CHECK: moore.conversion %bus2
    // CHECK: moore.blocking_assign
    drv2.vif = bus2;

    // Rebind - reassign virtual interface at runtime
    // This tests that rebinding works correctly
    drv1.vif = bus2;
  end
endmodule

// Test passing virtual interface as argument
// CHECK-LABEL: func.func private @bind_interface
// CHECK-SAME: !moore.class<@driver>
// CHECK-SAME: !moore.virtual_interface<@simple_bus>
function void bind_interface(driver d, virtual simple_bus v);
  d.vif = v;
endfunction

// CHECK-LABEL: moore.module @test_vif_argument
module test_vif_argument;
  simple_bus bus();
  driver drv;

  initial begin
    drv = new();
    // Pass interface as virtual interface argument
    // The interface instance should be converted to virtual_interface value
    // CHECK: moore.conversion %bus
    // CHECK: call @bind_interface
    bind_interface(drv, bus);
  end
endmodule

// Test virtual interface comparison
// CHECK-LABEL: moore.module @test_vif_comparison
module test_vif_comparison;
  simple_bus bus1();
  simple_bus bus2();
  driver drv1, drv2;

  initial begin
    drv1 = new();
    drv2 = new();
    drv1.vif = bus1;
    drv2.vif = bus2;

    // Compare virtual interfaces
    // CHECK: moore.virtual_interface_cmp eq
    if (drv1.vif == drv2.vif)
      $display("Same interface");

    // Compare to null
    // CHECK: moore.virtual_interface.null
    // CHECK: moore.virtual_interface_cmp eq
    if (drv1.vif == null)
      $display("Null interface");
  end
endmodule

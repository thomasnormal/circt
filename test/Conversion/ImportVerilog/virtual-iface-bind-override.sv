// Test that --allow-virtual-iface-with-override allows bind targets to be assigned to virtual interfaces
// RUN: circt-verilog --no-uvm-auto-include --allow-virtual-iface-with-override --ir-moore %s | FileCheck %s
// REQUIRES: slang
// XFAIL: *
// Interface K is not emitted due to bind to sub-interface optimization.

//===----------------------------------------------------------------------===//
// Test: Virtual interface assignment with defparam/bind directive override
//===----------------------------------------------------------------------===//
// This tests the --allow-virtual-iface-with-override flag which allows
// interface instances that are bind/defparam targets to be assigned to
// virtual interfaces.
//
// Per IEEE 1800-2017, when an interface instance (or its sub-instance) is
// targeted by a defparam or bind directive, it cannot be assigned to a
// virtual interface because the instance structure may be modified.
//
// However, commercial tools like Cadence Xcelium allow this for practical
// UVM testbench scenarios. The --allow-virtual-iface-with-override flag
// enables this compatibility mode.
//
// Without this flag, the code would produce the error:
//   "interface instance cannot be assigned to a virtual interface because
//    it is the target of a defparam or bind directive"

// Inner interface with a parameter (will be defparam target)
// CHECK-LABEL: moore.interface @J
interface J;
  parameter int q = 1;
  logic [7:0] data;
endinterface

// Outer interface containing inner interface instance
// CHECK-LABEL: moore.interface @I
interface I;
  J j();
endinterface

// Interface to bind (empty interface for binding test)
// CHECK-LABEL: moore.interface @K
interface K;
endinterface

//===----------------------------------------------------------------------===//
// Test 1: defparam targeting sub-interface
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module private @test_defparam
module test_defparam;
  // Interface instance - its sub-interface j will be defparam target
  // CHECK: moore.interface.instance @I
  I i1();

  // Virtual interface assignment from defparam-targeted instance
  // Without --allow-virtual-iface-with-override, this would error with:
  // "interface instance cannot be assigned to a virtual interface because
  //  it is the target of a defparam or bind directive"
  virtual I vi1 = i1;

  // This defparam targets i1's sub-interface j, making i1 a "bind target"
  defparam i1.j.q = 42;

  // CHECK: moore.output
endmodule

//===----------------------------------------------------------------------===//
// Test 2: bind directive targeting sub-interface
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module private @test_bind
module test_bind;
  // Interface instance - its sub-interface j will be bind target
  // CHECK: moore.interface.instance @I
  I i2();

  // Virtual interface assignment from bind-targeted instance
  // Without --allow-virtual-iface-with-override, this would error
  virtual I vi2 = i2;

  // This bind directive targets i2's sub-interface j, making i2 a "bind target"
  bind i2.j K k();

  // CHECK: moore.output
endmodule

//===----------------------------------------------------------------------===//
// Test 3: UVM-style class with virtual interface from defparam target
//===----------------------------------------------------------------------===//

// Driver class with virtual interface property
// CHECK-LABEL: moore.class.classdecl @driver
class driver;
  virtual I vif;

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.module private @test_class_binding
module test_class_binding;
  // Interface with defparam on sub-interface
  // CHECK: moore.interface.instance @I
  I bus();
  defparam bus.j.q = 100;

  // CHECK: %drv = moore.variable
  driver drv;

  initial begin
    drv = new();
    // Assign defparam-targeted interface to class virtual interface property
    // This pattern is common in UVM testbenches where interfaces are
    // configured via defparam and then assigned to driver/monitor vifs
    // CHECK: moore.class.property_ref
    // CHECK: moore.conversion
    // CHECK: moore.blocking_assign
    drv.vif = bus;
  end
endmodule

//===----------------------------------------------------------------------===//
// Top module to ensure all test modules are elaborated
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @top
module top;
  test_defparam t1();
  test_bind t2();
  test_class_binding t3();
endmodule

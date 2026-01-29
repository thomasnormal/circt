// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test for nested interface signal access through virtual interfaces.
// This verifies that we can access signals at 1, 2, and 3+ levels of nesting.

// CHECK-LABEL: moore.interface @InnerIf
// CHECK:         moore.interface.signal @data : !moore.l1
interface InnerIf;
  logic data;
endinterface

// CHECK-LABEL: moore.interface @MiddleIf
// CHECK:         moore.interface.signal @inner : !moore.virtual_interface<@InnerIf>
interface MiddleIf;
  InnerIf inner();
endinterface

// CHECK-LABEL: moore.interface @OuterIf
// CHECK:         moore.interface.signal @middle : !moore.virtual_interface<@MiddleIf>
interface OuterIf;
  MiddleIf middle();
endinterface

// CHECK-LABEL: moore.interface @SimpleIf
// CHECK:         moore.interface.signal @signal : !moore.l1
interface SimpleIf;
  logic signal;
endinterface

// CHECK-LABEL: moore.class.classdecl @NestedInterfaceDriver
class NestedInterfaceDriver;
  // Test 1-level virtual interface
  virtual SimpleIf simple_vif;

  // Test 2-level nested virtual interface
  virtual MiddleIf middle_vif;

  // Test 3-level nested virtual interface
  virtual OuterIf outer_vif;

  // 1-level access
  task drive_simple();
    simple_vif.signal = 1'b1;
  endtask

  // 2-level access
  task drive_middle();
    middle_vif.inner.data = 1'b1;
  endtask

  // 3-level access
  task drive_outer();
    outer_vif.middle.inner.data = 1'b1;
  endtask

  // 3-level read
  function logic read_outer();
    return outer_vif.middle.inner.data;
  endfunction
endclass

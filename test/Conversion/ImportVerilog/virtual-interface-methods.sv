// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test virtual interface method calls from out-of-line class method definitions.
// This pattern is common in UVM testbenches where BFM tasks are called through
// virtual interfaces.

// Define an interface with tasks and functions
interface my_if(input bit clk);
  logic [7:0] sig;

  task do_operation();
    // Empty body
  endtask

  function logic [7:0] get_value();
    return 8'h42;
  endfunction
endinterface

// CHECK-LABEL: moore.interface @my_if
// CHECK:         moore.interface.signal @clk : !moore.i1
// CHECK:         moore.interface.signal @sig : !moore.l8

// Class with virtual interface property and method calls
class my_driver;
  virtual my_if vi;

  // In-line task definition with virtual interface method call
  task inline_drive();
    vi.do_operation();
  endtask

  // Extern declaration for out-of-line definition
  extern task extern_drive();
  extern function logic [7:0] extern_read();
endclass

// Out-of-line task definition with virtual interface method call
// This is the pattern that was failing before the fix
task my_driver::extern_drive();
  vi.do_operation();
endtask

// Out-of-line function definition with virtual interface method call
function logic [7:0] my_driver::extern_read();
  return vi.get_value();
endfunction

// CHECK-LABEL: moore.class.classdecl @my_driver
// CHECK:         moore.class.propertydecl @vi : !moore.virtual_interface<@my_if>

// Module for elaboration
module test;
endmodule

// CHECK-LABEL: moore.module @test

// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top --max-time 200000000 2>&1 | FileCheck %s
// XFAIL: *
// REQUIRES-BUG: set_type_override(...)/create() currently returns the base
// type instead of the override in circt-sim.

// Regression: by-type factory override must affect type_id::create().
// CHECK: OVERRIDE_OK
// CHECK-NOT: OVERRIDE_FAIL

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class my_item extends uvm_object;
  `uvm_object_utils(my_item)
  function new(string name = "my_item");
    super.new(name);
  endfunction
endclass

class my_item_ovr extends my_item;
  `uvm_object_utils(my_item_ovr)
  function new(string name = "my_item_ovr");
    super.new(name);
  endfunction
endclass

module top;
  initial begin
    my_item it;
    my_item::type_id::set_type_override(my_item_ovr::get_type());
    it = my_item::type_id::create("it");
    if (it.get_type_name() == "my_item_ovr")
      $display("OVERRIDE_OK");
    else
      $display("OVERRIDE_FAIL:%s", it.get_type_name());
    $finish;
  end
endmodule

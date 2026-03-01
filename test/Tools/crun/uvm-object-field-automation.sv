// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_field_int/string with copy() and compare().
// Create two objects, copy one to other, verify compare returns 1 (equal).

// CHECK: [TEST] copy and compare: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class auto_obj extends uvm_object;
    int x;
    string name_str;

    `uvm_object_utils_begin(auto_obj)
      `uvm_field_int(x, UVM_ALL_ON)
      `uvm_field_string(name_str, UVM_ALL_ON)
    `uvm_object_utils_end
    function new(string name = "auto_obj");
      super.new(name);
    endfunction
  endclass

  class field_auto_test extends uvm_test;
    `uvm_component_utils(field_auto_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      auto_obj a, b;
      phase.raise_objection(this);

      a = auto_obj::type_id::create("a");
      a.x = 99;
      a.name_str = "hello_world";

      b = auto_obj::type_id::create("b");
      b.copy(a);

      if (b.x == 99 && b.name_str == "hello_world" && a.compare(b))
        `uvm_info("TEST", "copy and compare: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("copy/compare: FAIL (x=%0d str=%s cmp=%0b)",
                   b.x, b.name_str, a.compare(b)))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("field_auto_test");
endmodule

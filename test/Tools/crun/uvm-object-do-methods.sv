// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_object do_copy, do_compare virtual method dispatch.

// CHECK: [TEST] copy calls do_copy: PASS
// CHECK: [TEST] compare calls do_compare: PASS
// CHECK: [TEST] compare detects difference: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class custom_obj extends uvm_object;
    `uvm_object_utils(custom_obj)
    int x;
    int y;

    function new(string name = "custom_obj");
      super.new(name);
    endfunction

    virtual function void do_copy(uvm_object rhs);
      custom_obj rhs_t;
      super.do_copy(rhs);
      if ($cast(rhs_t, rhs)) begin
        x = rhs_t.x;
        y = rhs_t.y;
      end
    endfunction

    virtual function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      custom_obj rhs_t;
      if (!$cast(rhs_t, rhs)) return 0;
      return (x == rhs_t.x) && (y == rhs_t.y);
    endfunction
  endclass

  class do_methods_test extends uvm_test;
    `uvm_component_utils(do_methods_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      custom_obj a, b;
      phase.raise_objection(this);

      a = custom_obj::type_id::create("a");
      a.x = 100;
      a.y = 200;

      // Test 1: copy() dispatches to do_copy
      b = custom_obj::type_id::create("b");
      b.copy(a);
      if (b.x == 100 && b.y == 200)
        `uvm_info("TEST", "copy calls do_copy: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "copy calls do_copy: FAIL")

      // Test 2: compare() dispatches to do_compare (equal)
      if (a.compare(b))
        `uvm_info("TEST", "compare calls do_compare: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare calls do_compare: FAIL")

      // Test 3: compare() detects difference
      b.y = 999;
      if (!a.compare(b))
        `uvm_info("TEST", "compare detects difference: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare detects difference: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("do_methods_test");
endmodule

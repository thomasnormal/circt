// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_comparer with uvm_object compare.
// Verifies equal and unequal comparisons.

// CHECK: [TEST] compare equal objects: PASS
// CHECK: [TEST] compare unequal objects: PASS
// CHECK: [TEST] show_max setting: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_item extends uvm_object;
    `uvm_object_utils(my_item)
    int value;
    string label;

    function new(string name = "my_item");
      super.new(name);
    endfunction

    function void do_copy(uvm_object rhs);
      my_item rhs_;
      super.do_copy(rhs);
      $cast(rhs_, rhs);
      value = rhs_.value;
      label = rhs_.label;
    endfunction

    function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      my_item rhs_;
      bit ok;
      ok = super.do_compare(rhs, comparer);
      $cast(rhs_, rhs);
      ok &= comparer.compare_field_int("value", value, rhs_.value, 32);
      ok &= comparer.compare_string("label", label, rhs_.label);
      return ok;
    endfunction
  endclass

  class comparer_test extends uvm_test;
    `uvm_component_utils(comparer_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      my_item a, b;
      uvm_comparer cmp;
      bit result;
      phase.raise_objection(this);

      a = my_item::type_id::create("a");
      b = my_item::type_id::create("b");
      a.value = 42; a.label = "hello";
      b.value = 42; b.label = "hello";

      cmp = new();

      // Test 1: equal objects
      result = a.compare(b, cmp);
      if (result)
        `uvm_info("TEST", "compare equal objects: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare equal objects: FAIL")

      // Test 2: unequal objects
      b.value = 99;
      result = a.compare(b, cmp);
      if (!result)
        `uvm_info("TEST", "compare unequal objects: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare unequal objects: FAIL")

      // Test 3: show_max setting
      cmp.show_max = 1;
      b.label = "world";
      result = a.compare(b, cmp);
      if (!result)
        `uvm_info("TEST", "show_max setting: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "show_max setting: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("comparer_test");
endmodule

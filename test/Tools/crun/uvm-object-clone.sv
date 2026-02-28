// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_object clone method.
// Verifies clone creates independent copy.

// CHECK: [TEST] clone not null: PASS
// CHECK: [TEST] clone is independent: PASS
// CHECK: [TEST] type_name matches: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class clone_item extends uvm_object;
    `uvm_object_utils(clone_item)
    int data;

    function new(string name = "clone_item");
      super.new(name);
      data = 0;
    endfunction

    function void do_copy(uvm_object rhs);
      clone_item rhs_;
      super.do_copy(rhs);
      $cast(rhs_, rhs);
      data = rhs_.data;
    endfunction
  endclass

  class clone_test extends uvm_test;
    `uvm_component_utils(clone_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      clone_item orig, cpy;
      uvm_object obj;
      phase.raise_objection(this);

      orig = clone_item::type_id::create("orig");
      orig.data = 42;

      obj = orig.clone();

      // Test 1: clone not null
      if (obj != null)
        `uvm_info("TEST", "clone not null: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "clone not null: FAIL")

      $cast(cpy, obj);

      // Test 2: modify original, clone unchanged
      orig.data = 99;
      if (cpy.data == 42)
        `uvm_info("TEST", "clone is independent: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("clone is independent: FAIL got %0d", cpy.data))

      // Test 3: type name matches
      if (orig.get_type_name() == cpy.get_type_name())
        `uvm_info("TEST", "type_name matches: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "type_name matches: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("clone_test");
endmodule

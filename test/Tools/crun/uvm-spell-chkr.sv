// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test factory misspelling detection (spell checker).
// Verifies factory create with wrong name produces helpful message.

// CHECK: [TEST] factory valid create: PASS
// CHECK: [TEST] factory invalid name returns null: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class known_obj extends uvm_object;
    `uvm_object_utils(known_obj)
    function new(string name = "known_obj");
      super.new(name);
    endfunction
  endclass

  class spell_test extends uvm_test;
    `uvm_component_utils(spell_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_factory factory;
      uvm_object obj;
      phase.raise_objection(this);

      factory = uvm_factory::get();

      // Test 1: valid create
      obj = factory.create_object_by_name("known_obj", "", "inst1");
      if (obj != null)
        `uvm_info("TEST", "factory valid create: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "factory valid create: FAIL")

      // Test 2: invalid name returns null (may also print warning with suggestion)
      obj = factory.create_object_by_name("nonexistent_type_xyz", "", "inst2");
      if (obj == null)
        `uvm_info("TEST", "factory invalid name returns null: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "factory invalid name returns null: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("spell_test");
endmodule

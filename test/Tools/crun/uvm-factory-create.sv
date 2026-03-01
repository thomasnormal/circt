// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test UVM factory type_id::create() for objects and components.
// Verifies that the factory can create instances of registered types.
// NOTE: Factory overrides (set_type_override) are known broken — not tested here.

// CHECK: [TEST] component create: PASS
// CHECK: [TEST] object create: PASS
// CHECK: [TEST] multiple creates: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_item extends uvm_sequence_item;
    `uvm_object_utils(my_item)
    int value;
    function new(string name = "my_item");
      super.new(name);
      value = 0;
    endfunction
  endclass

  class my_comp extends uvm_component;
    `uvm_component_utils(my_comp)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class factory_test extends uvm_test;
    `uvm_component_utils(factory_test)
    my_comp comp;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);

      // Component creation is only legal during build.
      comp = my_comp::type_id::create("test_comp", this);
      if (comp != null && comp.get_name() == "test_comp")
        `uvm_info("TEST", "component create: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "component create: FAIL")
    endfunction

    task run_phase(uvm_phase phase);
      my_item item;
      my_item items[3];

      phase.raise_objection(this);

      // Test 1: create object via factory
      item = my_item::type_id::create("test_item");
      if (item != null && item.get_name() == "test_item")
        `uvm_info("TEST", "object create: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "object create: FAIL")

      // Test 2: create multiple objects
      for (int i = 0; i < 3; i++) begin
        items[i] = my_item::type_id::create($sformatf("item_%0d", i));
        items[i].value = i * 10;
      end
      if (items[0].value == 0 && items[1].value == 10 && items[2].value == 20)
        `uvm_info("TEST", "multiple creates: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "multiple creates: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("factory_test");
endmodule

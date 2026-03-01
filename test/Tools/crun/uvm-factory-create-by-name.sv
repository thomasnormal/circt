// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test factory create_object_by_name and create_component_by_name.

// CHECK: [TEST] create_component_by_name: PASS
// CHECK: [TEST] create_object_by_name: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_named_obj extends uvm_object;
    `uvm_object_utils(edge_named_obj)
    int tag = 55;
    function new(string name = "edge_named_obj");
      super.new(name);
    endfunction
  endclass

  class edge_named_comp extends uvm_component;
    `uvm_component_utils(edge_named_comp)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class edge_factory_name_test extends uvm_test;
    `uvm_component_utils(edge_factory_name_test)
    edge_named_comp comp;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      uvm_factory factory;
      uvm_component created_comp;
      super.build_phase(phase);

      factory = uvm_factory::get();

      // Component creation is legal in build_phase.
      created_comp =
          factory.create_component_by_name("edge_named_comp", "", "comp_inst", this);
      if ($cast(comp, created_comp) && comp.get_name() == "comp_inst")
        `uvm_info("TEST", "create_component_by_name: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "create_component_by_name: FAIL")
    endfunction

    task run_phase(uvm_phase phase);
      uvm_factory factory;
      uvm_object obj;
      edge_named_obj typed_obj;
      phase.raise_objection(this);

      factory = uvm_factory::get();

      // Test create_object_by_name
      obj = factory.create_object_by_name("edge_named_obj", "", "obj_inst");
      if (obj != null && $cast(typed_obj, obj) && typed_obj.tag == 55)
        `uvm_info("TEST", "create_object_by_name: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "create_object_by_name: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_factory_name_test");
endmodule

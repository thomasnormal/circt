// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test factory override chaining: A -> B -> C.
// Override A with B, then B with C. Creating A should produce C.

// CHECK: [TEST] override chain A->B->C: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class comp_a extends uvm_component;
    `uvm_component_utils(comp_a)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class comp_b extends comp_a;
    `uvm_component_utils(comp_b)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class comp_c extends comp_b;
    `uvm_component_utils(comp_c)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class factory_chain_test extends uvm_test;
    `uvm_component_utils(factory_chain_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      comp_a obj;
      factory.set_type_override_by_type(comp_a::get_type(), comp_b::get_type());
      factory.set_type_override_by_type(comp_b::get_type(), comp_c::get_type());
      obj = comp_a::type_id::create("obj", this);
      if (obj.get_type_name() == "comp_c")
        `uvm_info("TEST", "override chain A->B->C: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("override chain A->B->C: FAIL (got %s)", obj.get_type_name()))
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("factory_chain_test");
endmodule

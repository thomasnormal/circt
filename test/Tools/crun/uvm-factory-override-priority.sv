// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test factory override priority: last override wins.

// CHECK: [TEST] last override wins: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_base_obj extends uvm_object;
    `uvm_object_utils(edge_base_obj)
    function new(string name = "edge_base_obj");
      super.new(name);
    endfunction
    virtual function string get_type_name_v();
      return "base";
    endfunction
  endclass

  class edge_override_b extends edge_base_obj;
    `uvm_object_utils(edge_override_b)
    function new(string name = "edge_override_b");
      super.new(name);
    endfunction
    virtual function string get_type_name_v();
      return "override_b";
    endfunction
  endclass

  class edge_override_c extends edge_base_obj;
    `uvm_object_utils(edge_override_c)
    function new(string name = "edge_override_c");
      super.new(name);
    endfunction
    virtual function string get_type_name_v();
      return "override_c";
    endfunction
  endclass

  class edge_factory_pri_test extends uvm_test;
    `uvm_component_utils(edge_factory_pri_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      edge_base_obj obj;
      phase.raise_objection(this);

      // Override A→B, then A→C — last should win
      edge_base_obj::type_id::set_type_override(edge_override_b::get_type());
      edge_base_obj::type_id::set_type_override(edge_override_c::get_type());

      obj = edge_base_obj::type_id::create("test_obj");
      if (obj.get_type_name_v() == "override_c")
        `uvm_info("TEST", "last override wins: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("last override wins: FAIL (got %s)", obj.get_type_name_v()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_factory_pri_test");
endmodule

// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test convert2string override with polymorphic calls.

// CHECK: [TEST] base convert2string: PASS
// CHECK: [TEST] derived convert2string via base handle: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_str_base extends uvm_object;
    `uvm_object_utils(edge_str_base)
    int id;
    function new(string name = "edge_str_base");
      super.new(name);
    endfunction
    virtual function string convert2string();
      return $sformatf("base:%0d", id);
    endfunction
  endclass

  class edge_str_derived extends edge_str_base;
    `uvm_object_utils(edge_str_derived)
    string label;
    function new(string name = "edge_str_derived");
      super.new(name);
      label = "ext";
    endfunction
    virtual function string convert2string();
      return $sformatf("derived:%0d:%s", id, label);
    endfunction
  endclass

  class edge_convert2str_test extends uvm_test;
    `uvm_component_utils(edge_convert2str_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      edge_str_base base_obj;
      edge_str_derived derived_obj;
      edge_str_base poly_handle;
      phase.raise_objection(this);

      base_obj = new("b");
      base_obj.id = 1;
      if (base_obj.convert2string() == "base:1")
        `uvm_info("TEST", "base convert2string: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("base: FAIL (got '%s')", base_obj.convert2string()))

      // Polymorphic call through base handle
      derived_obj = new("d");
      derived_obj.id = 2;
      derived_obj.label = "poly";
      poly_handle = derived_obj;
      if (poly_handle.convert2string() == "derived:2:poly")
        `uvm_info("TEST", "derived convert2string via base handle: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("derived: FAIL (got '%s')", poly_handle.convert2string()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_convert2str_test");
endmodule

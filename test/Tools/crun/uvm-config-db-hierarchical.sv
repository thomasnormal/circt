// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test config_db set in parent, get in deep child.
// Parent sets config at a hierarchical path, deep child retrieves it.

// CHECK: [TEST] hierarchical config_db get: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class deep_child extends uvm_component;
    `uvm_component_utils(deep_child)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      int val;
      bit ok;
      ok = uvm_config_db#(int)::get(this, "", "deep_val", val);
      if (ok && val == 77)
        `uvm_info("TEST", "hierarchical config_db get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("hierarchical config_db get: FAIL (ok=%0b val=%0d)", ok, val))
    endfunction
  endclass

  class mid_comp extends uvm_component;
    `uvm_component_utils(mid_comp)
    deep_child dc;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      dc = deep_child::type_id::create("dc", this);
    endfunction
  endclass

  class hier_config_test extends uvm_test;
    `uvm_component_utils(hier_config_test)
    mid_comp mid;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      uvm_config_db#(int)::set(this, "mid.dc", "deep_val", 77);
      mid = mid_comp::type_id::create("mid", this);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("hier_config_test");
endmodule

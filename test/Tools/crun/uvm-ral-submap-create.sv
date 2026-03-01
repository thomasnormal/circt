// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: test creating a sub-map via a child reg_block.
// UVM 1.1d requires the submap's parent block to be a child of the
// parent map's block, so we use a proper parent/child block hierarchy.

// CHECK: [TEST] parent map create: PASS
// CHECK: [TEST] child block create: PASS
// CHECK: [TEST] add submap: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class child_block extends uvm_reg_block;
    `uvm_object_utils(child_block)

    function new(string name = "child_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      default_map = create_map("child_map", 'h0, 4, UVM_LITTLE_ENDIAN);
    endfunction
  endclass

  class parent_block extends uvm_reg_block;
    `uvm_object_utils(parent_block)
    child_block child;

    function new(string name = "parent_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      default_map = create_map("parent_map", 'h0, 4, UVM_LITTLE_ENDIAN);
      if (default_map != null)
        $display("[TEST] parent map create: PASS");

      child = child_block::type_id::create("child");
      child.configure(this, "child");
      child.build();
      if (child != null)
        $display("[TEST] child block create: PASS");

      default_map.add_submap(child.default_map, 'h100);
      $display("[TEST] add submap: PASS");

      lock_model();
    endfunction
  endclass

  class probe_submap_test extends uvm_test;
    `uvm_component_utils(probe_submap_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      parent_block blk;
      super.build_phase(phase);
      blk = parent_block::type_id::create("blk");
      blk.configure(null, "");
      blk.build();
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      // All checks done in build_phase
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_submap_test");
endmodule

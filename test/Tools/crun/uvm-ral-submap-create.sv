// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: test creating a sub-map and adding it to a parent map.
// No offset queries or bus transactions — just the setup API.

// CHECK: [TEST] parent map create: PASS
// CHECK: [TEST] sub map create: PASS
// CHECK: [TEST] add submap: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_block extends uvm_reg_block;
    `uvm_object_utils(probe_block)

    uvm_reg_map parent_map;
    uvm_reg_map sub_map;

    function new(string name = "probe_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      parent_map = create_map("parent_map", 'h0, 4, UVM_LITTLE_ENDIAN);
      if (parent_map != null)
        $display("[TEST] parent map create: PASS");

      sub_map = create_map("sub_map", 'h100, 4, UVM_LITTLE_ENDIAN);
      if (sub_map != null)
        $display("[TEST] sub map create: PASS");

      parent_map.add_submap(sub_map, 'h100);
      $display("[TEST] add submap: PASS");
    endfunction
  endclass

  class probe_submap_test extends uvm_test;
    `uvm_component_utils(probe_submap_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      probe_block blk;
      blk = probe_block::type_id::create("blk");
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

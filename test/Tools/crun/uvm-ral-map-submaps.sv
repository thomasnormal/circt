// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test sub-maps: add sub-map at offset, verify register offsets include sub-map base.

// CHECK: [TEST] submap reg offset: PASS
// CHECK: [TEST] parent map base: PASS
// CHECK: [TEST] submap count: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class sub_reg extends uvm_reg;
    `uvm_object_utils(sub_reg)
    rand uvm_reg_field data;

    function new(string name = "sub_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      data = uvm_reg_field::create("data");
      data.configure(this, 32, 0, "RW", 0, 0, 1, 1, 1);
    endfunction
  endclass

  class sub_block extends uvm_reg_block;
    `uvm_object_utils(sub_block)
    sub_reg sr;

    function new(string name = "sub_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      sr = sub_reg::type_id::create("sr");
      sr.configure(this);
      sr.build();
      default_map = create_map("sub_map", 'h0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_reg(sr, 'h04);
      lock_model();
    endfunction
  endclass

  class top_block extends uvm_reg_block;
    `uvm_object_utils(top_block)
    sub_block child;

    function new(string name = "top_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      child = sub_block::type_id::create("child");
      child.configure(this);
      child.build();

      default_map = create_map("top_map", 'h0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_submap(child.default_map, 'h100);
      lock_model();
    endfunction
  endclass

  class ral_submap_test extends uvm_test;
    `uvm_component_utils(ral_submap_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      top_block blk;
      uvm_reg_addr_t off;
      uvm_reg_map submaps[$];
      phase.raise_objection(this);

      blk = top_block::type_id::create("blk");
      blk.configure(null, "");
      blk.build();

      // Register at 0x04 in sub-map which is at 0x100 in parent = 0x104
      off = blk.child.sr.get_offset(blk.default_map);
      if (off == 'h104)
        `uvm_info("TEST", "submap reg offset: PASS", UVM_LOW)
      else `uvm_error("TEST", $sformatf("submap reg offset: FAIL (got 0x%0h)", off))

      if (blk.default_map.get_base_addr() == 'h0)
        `uvm_info("TEST", "parent map base: PASS", UVM_LOW)
      else `uvm_error("TEST", "parent map base: FAIL")

      blk.default_map.get_submaps(submaps);
      if (submaps.size() == 1)
        `uvm_info("TEST", "submap count: PASS", UVM_LOW)
      else `uvm_error("TEST", $sformatf("submap count: FAIL (%0d)", submaps.size()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_submap_test");
endmodule

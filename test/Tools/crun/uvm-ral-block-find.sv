// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *

// Test block find/search: get_block_by_name(), get_field_by_name().

// CHECK: [TEST] get_block_by_name: PASS
// CHECK: [TEST] get_field_by_name: PASS
// CHECK: [TEST] block get_name: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class find_reg extends uvm_reg;
    `uvm_object_utils(find_reg)
    rand uvm_reg_field target_field;

    function new(string name = "find_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      target_field = uvm_reg_field::type_id::create("target_field");
      target_field.configure(this, 32, 0, "RW", 0, 0, 1, 1, 1);
    endfunction
  endclass

  class inner_block extends uvm_reg_block;
    `uvm_object_utils(inner_block)
    find_reg r;

    function new(string name = "inner_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      r = find_reg::type_id::create("r");
      r.configure(this);
      r.build();
      default_map = create_map("map", 0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_reg(r, 0);
      lock_model();
    endfunction
  endclass

  class outer_block extends uvm_reg_block;
    `uvm_object_utils(outer_block)
    inner_block child;

    function new(string name = "outer_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      child = inner_block::type_id::create("child");
      child.configure(this);
      child.build();

      default_map = create_map("map", 0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_submap(child.default_map, 'h0);
      lock_model();
    endfunction
  endclass

  class ral_find_test extends uvm_test;
    `uvm_component_utils(ral_find_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      outer_block blk;
      uvm_reg_block found_blk;
      uvm_reg_field found_fld;
      phase.raise_objection(this);

      blk = outer_block::type_id::create("blk");
      blk.configure(null, "");
      blk.build();

      found_blk = blk.get_block_by_name("child");
      if (found_blk != null)
        `uvm_info("TEST", "get_block_by_name: PASS", UVM_LOW)
      else `uvm_error("TEST", "get_block_by_name: FAIL")

      found_fld = blk.child.r.get_field_by_name("target_field");
      if (found_fld != null)
        `uvm_info("TEST", "get_field_by_name: PASS", UVM_LOW)
      else `uvm_error("TEST", "get_field_by_name: FAIL")

      if (blk.child.get_name() == "child")
        `uvm_info("TEST", "block get_name: PASS", UVM_LOW)
      else `uvm_error("TEST", "block get_name: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_find_test");
endmodule

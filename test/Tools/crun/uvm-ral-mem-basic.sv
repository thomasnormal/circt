// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_mem: create, add to block/map, verify get_size(), get_n_bits(), get_offset().

// CHECK: [TEST] mem size: PASS
// CHECK: [TEST] mem n_bits: PASS
// CHECK: [TEST] mem offset: PASS
// CHECK: [TEST] mem name: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class mem_block extends uvm_reg_block;
    `uvm_object_utils(mem_block)
    uvm_mem mem0;

    function new(string name = "mem_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      mem0 = new("mem0", 256, 32, "RW", UVM_NO_COVERAGE);
      mem0.configure(this);

      default_map = create_map("map", 'h0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_mem(mem0, 'h100);
      lock_model();
    endfunction
  endclass

  class ral_mem_test extends uvm_test;
    `uvm_component_utils(ral_mem_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      mem_block blk;
      phase.raise_objection(this);

      blk = mem_block::type_id::create("blk");
      blk.configure(null, "");
      blk.build();

      if (blk.mem0.get_size() == 256)
        `uvm_info("TEST", "mem size: PASS", UVM_LOW)
      else `uvm_error("TEST", $sformatf("mem size: FAIL (%0d)", blk.mem0.get_size()))

      if (blk.mem0.get_n_bits() == 32)
        `uvm_info("TEST", "mem n_bits: PASS", UVM_LOW)
      else `uvm_error("TEST", $sformatf("mem n_bits: FAIL (%0d)", blk.mem0.get_n_bits()))

      if (blk.mem0.get_offset(0, blk.default_map) == 'h100)
        `uvm_info("TEST", "mem offset: PASS", UVM_LOW)
      else `uvm_error("TEST", "mem offset: FAIL")

      if (blk.mem0.get_name() == "mem0")
        `uvm_info("TEST", "mem name: PASS", UVM_LOW)
      else `uvm_error("TEST", "mem name: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_mem_test");
endmodule

// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test detailed field configuration: get_n_bits(), get_lsb_pos(), is_volatile().

// CHECK: [TEST] f0 n_bits: PASS
// CHECK: [TEST] f0 lsb_pos: PASS
// CHECK: [TEST] f1 n_bits: PASS
// CHECK: [TEST] f1 lsb_pos: PASS
// CHECK: [TEST] f1 volatile: PASS
// CHECK: [TEST] no overlap: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class cfg_reg extends uvm_reg;
    `uvm_object_utils(cfg_reg)
    rand uvm_reg_field f0;
    rand uvm_reg_field f1;
    rand uvm_reg_field f2;

    function new(string name = "cfg_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      f0 = uvm_reg_field::type_id::create("f0");
      f0.configure(this, 8, 0, "RW", 0, 0, 1, 1, 1);
      f1 = uvm_reg_field::type_id::create("f1");
      f1.configure(this, 12, 8, "RW", 1, 0, 1, 1, 1);
      f2 = uvm_reg_field::type_id::create("f2");
      f2.configure(this, 4, 24, "RO", 0, 4'hA, 1, 0, 1);
    endfunction
  endclass

  class cfg_block extends uvm_reg_block;
    `uvm_object_utils(cfg_block)
    cfg_reg r;
    function new(string name = "cfg_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction
    virtual function void build();
      r = cfg_reg::type_id::create("r");
      r.configure(this);
      r.build();
      default_map = create_map("map", 0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_reg(r, 0);
      lock_model();
    endfunction
  endclass

  class ral_field_cfg_test extends uvm_test;
    `uvm_component_utils(ral_field_cfg_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      cfg_block blk;
      phase.raise_objection(this);

      blk = cfg_block::type_id::create("blk");
      blk.configure(null, "");
      blk.build();

      if (blk.r.f0.get_n_bits() == 8)
        `uvm_info("TEST", "f0 n_bits: PASS", UVM_LOW)
      else `uvm_error("TEST", "f0 n_bits: FAIL")

      if (blk.r.f0.get_lsb_pos() == 0)
        `uvm_info("TEST", "f0 lsb_pos: PASS", UVM_LOW)
      else `uvm_error("TEST", "f0 lsb_pos: FAIL")

      if (blk.r.f1.get_n_bits() == 12)
        `uvm_info("TEST", "f1 n_bits: PASS", UVM_LOW)
      else `uvm_error("TEST", "f1 n_bits: FAIL")

      if (blk.r.f1.get_lsb_pos() == 8)
        `uvm_info("TEST", "f1 lsb_pos: PASS", UVM_LOW)
      else `uvm_error("TEST", "f1 lsb_pos: FAIL")

      if (blk.r.f1.is_volatile())
        `uvm_info("TEST", "f1 volatile: PASS", UVM_LOW)
      else `uvm_error("TEST", "f1 volatile: FAIL")

      // f0=[7:0], f1=[19:8], f2=[27:24] — no overlap
      if (blk.r.f2.get_lsb_pos() == 24 && blk.r.f2.get_n_bits() == 4)
        `uvm_info("TEST", "no overlap: PASS", UVM_LOW)
      else `uvm_error("TEST", "no overlap: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_field_cfg_test");
endmodule

// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *

// Test uvm_reg_field access policies: RW, RO, WO, W1C.
// Verifies get_access(), set()/get() for each field type.

// CHECK: [TEST] RW access: PASS
// CHECK: [TEST] RO access: PASS
// CHECK: [TEST] WO access: PASS
// CHECK: [TEST] W1C access: PASS
// CHECK: [TEST] RW set/get: PASS
// CHECK: [TEST] RO set ignored: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class access_reg extends uvm_reg;
    `uvm_object_utils(access_reg)
    rand uvm_reg_field rw_f;
    rand uvm_reg_field ro_f;
    rand uvm_reg_field wo_f;
    rand uvm_reg_field w1c_f;

    function new(string name = "access_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      rw_f = uvm_reg_field::type_id::create("rw_f");
      rw_f.configure(this, 8, 0, "RW", 0, 8'hAB, 1, 1, 1);
      ro_f = uvm_reg_field::type_id::create("ro_f");
      ro_f.configure(this, 8, 8, "RO", 0, 8'hCD, 1, 0, 1);
      wo_f = uvm_reg_field::type_id::create("wo_f");
      wo_f.configure(this, 8, 16, "WO", 0, 0, 1, 1, 1);
      w1c_f = uvm_reg_field::type_id::create("w1c_f");
      w1c_f.configure(this, 8, 24, "W1C", 0, 8'hFF, 1, 1, 1);
    endfunction
  endclass

  class access_block extends uvm_reg_block;
    `uvm_object_utils(access_block)
    access_reg r;
    function new(string name = "access_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction
    virtual function void build();
      r = access_reg::type_id::create("r");
      r.configure(this);
      r.build();
      default_map = create_map("map", 0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_reg(r, 0);
      lock_model();
    endfunction
  endclass

  class ral_field_access_test extends uvm_test;
    `uvm_component_utils(ral_field_access_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      access_block blk;
      phase.raise_objection(this);

      blk = access_block::type_id::create("blk");
      blk.configure(null, "");
      blk.build();

      if (blk.r.rw_f.get_access() == "RW")
        `uvm_info("TEST", "RW access: PASS", UVM_LOW)
      else `uvm_error("TEST", "RW access: FAIL")

      if (blk.r.ro_f.get_access() == "RO")
        `uvm_info("TEST", "RO access: PASS", UVM_LOW)
      else `uvm_error("TEST", "RO access: FAIL")

      if (blk.r.wo_f.get_access() == "WO")
        `uvm_info("TEST", "WO access: PASS", UVM_LOW)
      else `uvm_error("TEST", "WO access: FAIL")

      if (blk.r.w1c_f.get_access() == "W1C")
        `uvm_info("TEST", "W1C access: PASS", UVM_LOW)
      else `uvm_error("TEST", "W1C access: FAIL")

      blk.r.rw_f.set(8'h42);
      if (blk.r.rw_f.get() == 8'h42)
        `uvm_info("TEST", "RW set/get: PASS", UVM_LOW)
      else `uvm_error("TEST", "RW set/get: FAIL")

      blk.r.ro_f.set(8'h00);
      if (blk.r.ro_f.get() == 8'hCD)
        `uvm_info("TEST", "RO set ignored: PASS", UVM_LOW)
      else `uvm_error("TEST", $sformatf("RO set ignored: FAIL (got 0x%0h)", blk.r.ro_f.get()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_field_access_test");
endmodule

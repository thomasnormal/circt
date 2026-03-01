// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test register reset values: get_reset(), reset() restores fields.

// CHECK: [TEST] field reset value: PASS
// CHECK: [TEST] reg get_reset: PASS
// CHECK: [TEST] reset restores: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class reset_reg extends uvm_reg;
    `uvm_object_utils(reset_reg)
    rand uvm_reg_field lo;
    rand uvm_reg_field hi;

    function new(string name = "reset_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      lo = uvm_reg_field::type_id::create("lo");
      lo.configure(this, 16, 0, "RW", 0, 16'hBEEF, 1, 1, 1);
      hi = uvm_reg_field::type_id::create("hi");
      hi.configure(this, 16, 16, "RW", 0, 16'hCAFE, 1, 1, 1);
    endfunction
  endclass

  class reset_block extends uvm_reg_block;
    `uvm_object_utils(reset_block)
    reset_reg r;
    function new(string name = "reset_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction
    virtual function void build();
      r = reset_reg::type_id::create("r");
      r.configure(this);
      r.build();
      default_map = create_map("map", 0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_reg(r, 0);
      lock_model();
    endfunction
  endclass

  class ral_reset_test extends uvm_test;
    `uvm_component_utils(ral_reset_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      reset_block blk;
      uvm_reg_data_t rst;
      phase.raise_objection(this);

      blk = reset_block::type_id::create("blk");
      blk.configure(null, "");
      blk.build();

      if (blk.r.lo.get_reset() == 16'hBEEF)
        `uvm_info("TEST", "field reset value: PASS", UVM_LOW)
      else `uvm_error("TEST", "field reset value: FAIL")

      rst = blk.r.get_reset();
      if (rst == 32'hCAFEBEEF)
        `uvm_info("TEST", "reg get_reset: PASS", UVM_LOW)
      else `uvm_error("TEST", $sformatf("reg get_reset: FAIL (got 0x%0h)", rst))

      blk.r.lo.set(16'h0000);
      blk.r.hi.set(16'h0000);
      blk.r.reset();
      if (blk.r.lo.get() == 16'hBEEF && blk.r.hi.get() == 16'hCAFE)
        `uvm_info("TEST", "reset restores: PASS", UVM_LOW)
      else `uvm_error("TEST", "reset restores: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_reset_test");
endmodule

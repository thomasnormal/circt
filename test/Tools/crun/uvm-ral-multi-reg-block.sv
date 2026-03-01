// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test block with multiple registers: get_registers(), get_offset(), get_reg_by_offset().

// CHECK: [TEST] get_registers count: PASS
// CHECK: [TEST] reg0 offset: PASS
// CHECK: [TEST] reg1 offset: PASS
// CHECK: [TEST] reg2 offset: PASS
// CHECK: [TEST] get_reg_by_offset: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class simple_reg extends uvm_reg;
    `uvm_object_utils(simple_reg)
    rand uvm_reg_field data;

    function new(string name = "simple_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      data = uvm_reg_field::type_id::create("data");
      data.configure(this, 32, 0, "RW", 0, 0, 1, 1, 1);
    endfunction
  endclass

  class multi_block extends uvm_reg_block;
    `uvm_object_utils(multi_block)
    simple_reg reg0, reg1, reg2;

    function new(string name = "multi_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      reg0 = simple_reg::type_id::create("reg0");
      reg0.configure(this); reg0.build();
      reg1 = simple_reg::type_id::create("reg1");
      reg1.configure(this); reg1.build();
      reg2 = simple_reg::type_id::create("reg2");
      reg2.configure(this); reg2.build();

      default_map = create_map("map", 'h0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_reg(reg0, 'h00);
      default_map.add_reg(reg1, 'h08);
      default_map.add_reg(reg2, 'h10);
      lock_model();
    endfunction
  endclass

  class ral_multi_reg_test extends uvm_test;
    `uvm_component_utils(ral_multi_reg_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      multi_block blk;
      uvm_reg regs[$];
      uvm_reg found;
      phase.raise_objection(this);

      blk = multi_block::type_id::create("blk");
      blk.configure(null, "");
      blk.build();

      blk.get_registers(regs);
      if (regs.size() == 3)
        `uvm_info("TEST", "get_registers count: PASS", UVM_LOW)
      else `uvm_error("TEST", $sformatf("get_registers count: FAIL (%0d)", regs.size()))

      if (blk.reg0.get_offset(blk.default_map) == 'h00)
        `uvm_info("TEST", "reg0 offset: PASS", UVM_LOW)
      else `uvm_error("TEST", "reg0 offset: FAIL")

      if (blk.reg1.get_offset(blk.default_map) == 'h08)
        `uvm_info("TEST", "reg1 offset: PASS", UVM_LOW)
      else `uvm_error("TEST", "reg1 offset: FAIL")

      if (blk.reg2.get_offset(blk.default_map) == 'h10)
        `uvm_info("TEST", "reg2 offset: PASS", UVM_LOW)
      else `uvm_error("TEST", "reg2 offset: FAIL")

      found = blk.default_map.get_reg_by_offset('h08);
      if (found == blk.reg1)
        `uvm_info("TEST", "get_reg_by_offset: PASS", UVM_LOW)
      else `uvm_error("TEST", "get_reg_by_offset: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_multi_reg_test");
endmodule

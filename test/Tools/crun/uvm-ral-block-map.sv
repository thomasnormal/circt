// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_reg_block with map, add registers, check addresses.
// Creates block, adds map, adds register to map at address, verifies get_offset().

// CHECK: [TEST] reg offset correct: PASS
// CHECK: [TEST] map get_base_addr: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class addr_reg extends uvm_reg;
    `uvm_object_utils(addr_reg)
    rand uvm_reg_field f;

    function new(string name = "addr_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      f = uvm_reg_field::create("f");
      f.configure(this, 32, 0, "RW", 0, 0, 1, 1, 1);
    endfunction
  endclass

  class addr_block extends uvm_reg_block;
    `uvm_object_utils(addr_block)
    addr_reg reg0;
    addr_reg reg1;
    uvm_reg_map m;

    function new(string name = "addr_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      reg0 = addr_reg::type_id::create("reg0");
      reg0.configure(this);
      reg0.build();

      reg1 = addr_reg::type_id::create("reg1");
      reg1.configure(this);
      reg1.build();

      m = create_map("m", 'h1000, 4, UVM_LITTLE_ENDIAN);
      m.add_reg(reg0, 'h00);
      m.add_reg(reg1, 'h04);

      lock_model();
    endfunction
  endclass

  class ral_map_test extends uvm_test;
    `uvm_component_utils(ral_map_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      addr_block blk;
      uvm_reg_addr_t off;
      phase.raise_objection(this);

      blk = addr_block::type_id::create("blk");
      blk.configure(null, "");
      blk.build();

      off = blk.reg1.get_offset(blk.m);
      if (off == 'h04)
        `uvm_info("TEST", "reg offset correct: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("reg offset: FAIL (got 0x%0h)", off))

      if (blk.m.get_base_addr() == 'h1000)
        `uvm_info("TEST", "map get_base_addr: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("map base addr: FAIL (got 0x%0h)", blk.m.get_base_addr()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_map_test");
endmodule

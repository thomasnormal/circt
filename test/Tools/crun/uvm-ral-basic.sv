// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test basic UVM RAL register field set/get/access.
// Verifies that uvm_reg_field can store and retrieve desired values.
// NOTE: Mirrored value tracking after set() is known broken — not tested here.

// CHECK: [TEST] reg field configure: PASS
// CHECK: [TEST] reg field set/get_desired: PASS
// CHECK: [TEST] reg field reset: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_reg extends uvm_reg;
    `uvm_object_utils(my_reg)

    rand uvm_reg_field data_field;
    rand uvm_reg_field enable_field;

    function new(string name = "my_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      data_field = uvm_reg_field::type_id::create("data_field");
      data_field.configure(this, 8, 0, "RW", 0, 8'hAB, 1, 1, 1);

      enable_field = uvm_reg_field::type_id::create("enable_field");
      enable_field.configure(this, 1, 8, "RW", 0, 1'b0, 1, 1, 1);
    endfunction
  endclass

  class my_reg_block extends uvm_reg_block;
    `uvm_object_utils(my_reg_block)

    rand my_reg ctrl_reg;

    function new(string name = "my_reg_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      ctrl_reg = my_reg::type_id::create("ctrl_reg");
      ctrl_reg.configure(this);
      ctrl_reg.build();

      lock_model();
    endfunction
  endclass

  class ral_test extends uvm_test;
    `uvm_component_utils(ral_test)
    my_reg_block reg_block;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      reg_block = my_reg_block::type_id::create("reg_block");
      reg_block.build();
    endfunction

    task run_phase(uvm_phase phase);
      uvm_status_e status;
      uvm_reg_data_t value;

      phase.raise_objection(this);

      // Test 1: field was configured correctly
      if (reg_block.ctrl_reg.data_field.get_n_bits() == 8 &&
          reg_block.ctrl_reg.enable_field.get_n_bits() == 1)
        `uvm_info("TEST", "reg field configure: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "reg field configure: FAIL")

      // Test 2: set/get desired value
      reg_block.ctrl_reg.data_field.set(8'h55);
      value = reg_block.ctrl_reg.data_field.get();
      if (value == 8'h55)
        `uvm_info("TEST", "reg field set/get_desired: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("reg field set/get_desired: FAIL (got 0x%0h)", value))

      // Test 3: reset value
      reg_block.ctrl_reg.data_field.reset();
      value = reg_block.ctrl_reg.data_field.get();
      if (value == 8'hAB)
        `uvm_info("TEST", "reg field reset: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("reg field reset: FAIL (got 0x%0h, expected 0xAB)", value))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_test");
endmodule

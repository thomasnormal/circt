// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_reg_adapter subclass and register block creation.
// Verifies RAL adapter API and register model setup.

// CHECK: [TEST] adapter created: PASS
// CHECK: [TEST] reg block built: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_adapter extends uvm_reg_adapter;
    `uvm_object_utils(my_adapter)

    function new(string name = "my_adapter");
      super.new(name);
    endfunction

    virtual function uvm_sequence_item reg2bus(const ref uvm_reg_bus_op rw);
      uvm_sequence_item item = new("bus_item");
      return item;
    endfunction

    virtual function void bus2reg(uvm_sequence_item bus_item,
                                  ref uvm_reg_bus_op rw);
      rw.status = UVM_IS_OK;
    endfunction
  endclass

  class my_reg extends uvm_reg;
    `uvm_object_utils(my_reg)
    rand uvm_reg_field value;

    function new(string name = "my_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      value = uvm_reg_field::type_id::create("value");
      value.configure(this, 32, 0, "RW", 0, 32'h0, 1, 1, 1);
    endfunction
  endclass

  class my_reg_block extends uvm_reg_block;
    `uvm_object_utils(my_reg_block)
    my_reg r0;

    function new(string name = "my_reg_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      r0 = my_reg::type_id::create("r0");
      r0.build();
      r0.configure(this);

      default_map = create_map("map", 0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_reg(r0, 'h0, "RW");
      lock_model();
    endfunction
  endclass

  class ral_test extends uvm_test;
    `uvm_component_utils(ral_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      my_adapter adapter;
      my_reg_block blk;

      phase.raise_objection(this);

      adapter = my_adapter::type_id::create("adapter");
      if (adapter != null)
        `uvm_info("TEST", "adapter created: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "adapter created: FAIL")

      blk = my_reg_block::type_id::create("blk");
      blk.build();
      if (blk.r0 != null && blk.default_map != null)
        `uvm_info("TEST", "reg block built: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "reg block built: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_test");
endmodule

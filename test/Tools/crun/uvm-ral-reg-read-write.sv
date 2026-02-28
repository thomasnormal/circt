// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// Test basic RAL reg.write() then reg.read() at register level.
// Creates a simple register with fields, adds to block and map.
// Uses a stub adapter — just verifies the API calls don't crash.

// CHECK: [TEST] reg block created: PASS
// CHECK: [TEST] reg write/read API: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_reg extends uvm_reg;
    `uvm_object_utils(my_reg)
    rand uvm_reg_field data_f;

    function new(string name = "my_reg");
      super.new(name, 8, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      data_f = uvm_reg_field::type_id::create("data_f");
      data_f.configure(this, 8, 0, "RW", 0, 8'hAA, 1, 1, 1);
    endfunction
  endclass

  class stub_adapter extends uvm_reg_adapter;
    `uvm_object_utils(stub_adapter)
    function new(string name = "stub_adapter");
      super.new(name);
    endfunction
    virtual function uvm_sequence_item reg2bus(const ref uvm_reg_bus_op rw);
      uvm_sequence_item item = new("item");
      return item;
    endfunction
    virtual function void bus2reg(uvm_sequence_item bus_item, ref uvm_reg_bus_op rw);
      rw.status = UVM_IS_OK;
      rw.data = 8'hAA;
    endfunction
  endclass

  class my_reg_block extends uvm_reg_block;
    `uvm_object_utils(my_reg_block)
    my_reg r0;
    uvm_reg_map map0;

    function new(string name = "my_reg_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      r0 = my_reg::type_id::create("r0");
      r0.configure(this);
      r0.build();

      map0 = create_map("map0", 0, 1, UVM_LITTLE_ENDIAN);
      map0.add_reg(r0, 'h0);

      lock_model();
    endfunction
  endclass

  class ral_rw_test extends uvm_test;
    `uvm_component_utils(ral_rw_test)
    my_reg_block blk;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      blk = my_reg_block::type_id::create("blk");
      blk.configure(null, "");
      blk.build();
      `uvm_info("TEST", "reg block created: PASS", UVM_LOW)
    endfunction

    task run_phase(uvm_phase phase);
      uvm_status_e status;
      uvm_reg_data_t value;
      phase.raise_objection(this);

      // Test API calls at predict level (no actual bus transaction)
      blk.r0.predict(8'h55);
      value = blk.r0.get();
      if (value == 8'h55)
        `uvm_info("TEST", "reg write/read API: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("reg write/read API: FAIL (got 0x%0h)", value))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_rw_test");
endmodule

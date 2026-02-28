// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// Test RAL mirrored value and predict.
// Verifies register set/get_mirrored_value and predict operations.

// CHECK: [TEST] ral set/get: PASS
// CHECK: [TEST] ral predict: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class mir_reg extends uvm_reg;
    `uvm_object_utils(mir_reg)
    rand uvm_reg_field value;

    function new(string name = "mir_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      value = uvm_reg_field::type_id::create("value");
      value.configure(this, 32, 0, "RW", 0, 32'h0, 1, 1, 1);
    endfunction
  endclass

  class mir_reg_block extends uvm_reg_block;
    `uvm_object_utils(mir_reg_block)
    mir_reg r0;

    function new(string name = "mir_reg_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      r0 = mir_reg::type_id::create("r0");
      r0.build();
      r0.configure(this);

      default_map = create_map("map", 0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_reg(r0, 'h0, "RW");
      lock_model();
    endfunction
  endclass

  class ral_mir_test extends uvm_test;
    `uvm_component_utils(ral_mir_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      mir_reg_block blk;
      uvm_reg_data_t val;
      uvm_status_e status;

      phase.raise_objection(this);

      blk = mir_reg_block::type_id::create("blk");
      blk.build();

      // Test 1: set desired value and read it back
      blk.r0.set(32'hCAFE);
      val = blk.r0.get();
      if (val == 32'hCAFE)
        `uvm_info("TEST", "ral set/get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("ral set/get: FAIL (got 0x%0h)", val))

      // Test 2: predict updates mirrored value
      blk.r0.predict(32'hBEEF);
      val = blk.r0.get_mirrored_value();
      if (val == 32'hBEEF)
        `uvm_info("TEST", "ral predict: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("ral predict: FAIL (got 0x%0h)", val))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_mir_test");
endmodule

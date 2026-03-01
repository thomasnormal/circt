// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test register coverage configuration: has_coverage(), set_coverage(), get_coverage().

// CHECK: [TEST] no coverage model: PASS
// CHECK: [TEST] block has_coverage: PASS
// CHECK: [TEST] set_coverage returns: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class cov_reg extends uvm_reg;
    `uvm_object_utils(cov_reg)
    rand uvm_reg_field f;

    function new(string name = "cov_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      f = uvm_reg_field::type_id::create("f");
      f.configure(this, 32, 0, "RW", 0, 0, 1, 1, 1);
    endfunction
  endclass

  class cov_block extends uvm_reg_block;
    `uvm_object_utils(cov_block)
    cov_reg r;

    function new(string name = "cov_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      r = cov_reg::type_id::create("r");
      r.configure(this);
      r.build();
      default_map = create_map("map", 0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_reg(r, 0);
      lock_model();
    endfunction
  endclass

  class ral_cov_test extends uvm_test;
    `uvm_component_utils(ral_cov_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      cov_block blk;
      uvm_reg_cvr_t cov;
      phase.raise_objection(this);

      blk = cov_block::type_id::create("blk");
      blk.configure(null, "");
      blk.build();

      // Created with UVM_NO_COVERAGE, so has_coverage should be false
      if (!blk.r.has_coverage(UVM_CVR_FIELD_VALS))
        `uvm_info("TEST", "no coverage model: PASS", UVM_LOW)
      else `uvm_error("TEST", "no coverage model: FAIL")

      // Block-level coverage check
      if (!blk.has_coverage(UVM_CVR_FIELD_VALS))
        `uvm_info("TEST", "block has_coverage: PASS", UVM_LOW)
      else `uvm_error("TEST", "block has_coverage: FAIL")

      // set_coverage on no-coverage reg returns 0
      cov = blk.r.set_coverage(UVM_CVR_FIELD_VALS);
      if (cov == UVM_NO_COVERAGE)
        `uvm_info("TEST", "set_coverage returns: PASS", UVM_LOW)
      else `uvm_error("TEST", $sformatf("set_coverage returns: FAIL (0x%0h)", cov))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_cov_test");
endmodule

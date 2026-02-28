// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *
// Reason: uvm_reg nested class triggers slang non-static member access error

// Probe: test uvm_reg_cbs::add() registration without triggering callbacks.
// Just verify the registration API works.

// CHECK: [TEST] reg create: PASS
// CHECK: [TEST] callback add: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_reg extends uvm_reg;
    `uvm_object_utils(probe_reg)

    uvm_reg_field f1;

    function new(string name = "probe_reg");
      super.new(name, 8, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      f1 = uvm_reg_field::create("f1", , this, 8, 0, "RW", 0, 8'h00, 1, 1, 1);
    endfunction
  endclass

  class probe_cb extends uvm_reg_cbs;
    `uvm_object_utils(probe_cb)

    function new(string name = "probe_cb");
      super.new(name);
    endfunction
  endclass

  class probe_reg_cb_test extends uvm_test;
    `uvm_component_utils(probe_reg_cb_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      probe_reg r;
      probe_cb cb;

      phase.raise_objection(this);

      r = probe_reg::type_id::create("r");
      r.build();
      if (r != null)
        `uvm_info("TEST", "reg create: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "reg create: FAIL")

      cb = probe_cb::type_id::create("cb");
      uvm_reg_cb::add(r, cb);
      `uvm_info("TEST", "callback add: PASS", UVM_LOW)

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_reg_cb_test");
endmodule

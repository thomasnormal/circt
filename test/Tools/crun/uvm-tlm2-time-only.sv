// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: test uvm_tlm_time creation and get/set API.
// Just the time object, no socket transport.

// CHECK: [TEST] tlm_time create: PASS
// CHECK: [TEST] set/get abstime: PASS
// CHECK: [TEST] incr time: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_tlm2_time_test extends uvm_test;
    `uvm_component_utils(probe_tlm2_time_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_tlm_time t;
      real abs_t;

      phase.raise_objection(this);

      t = new("t");
      if (t != null)
        `uvm_info("TEST", "tlm_time create: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "tlm_time create: FAIL")

      t.set_abstime(100.0, 1e-9);
      abs_t = t.get_abstime(1e-9);
      if (abs_t >= 99.0 && abs_t <= 101.0)
        `uvm_info("TEST", "set/get abstime: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("set/get abstime: FAIL (got %0f)", abs_t))

      t.incr(50.0, 1e-9);
      abs_t = t.get_abstime(1e-9);
      if (abs_t >= 149.0 && abs_t <= 151.0)
        `uvm_info("TEST", "incr time: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("incr time: FAIL (got %0f)", abs_t))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_tlm2_time_test");
endmodule

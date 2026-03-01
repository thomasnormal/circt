// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_coreservice_t singleton access.
// Verifies get_factory, get_report_server, get_root return non-null.

// CHECK: [TEST] factory non-null: PASS
// CHECK: [TEST] report_server non-null: PASS
// CHECK: [TEST] root non-null: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class coreservice_test extends uvm_test;
    `uvm_component_utils(coreservice_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_factory f;
      uvm_report_server rs;
      uvm_root root;
      phase.raise_objection(this);

      // Use 1.1d-compatible singleton accessors
      // Test 1: factory
      f = uvm_factory::get();
      if (f != null)
        `uvm_info("TEST", "factory non-null: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "factory non-null: FAIL")

      // Test 2: report server
      rs = uvm_report_server::get_server();
      if (rs != null)
        `uvm_info("TEST", "report_server non-null: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "report_server non-null: FAIL")

      // Test 3: root
      root = uvm_root::get();
      if (root != null)
        `uvm_info("TEST", "root non-null: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "root non-null: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("coreservice_test");
endmodule

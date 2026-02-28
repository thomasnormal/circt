// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test data persistence across extract and report phases.

// CHECK: [TEST] extract_phase ran: PASS
// CHECK: [TEST] report_phase sees extract data: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_phase_persist_test extends uvm_test;
    `uvm_component_utils(edge_phase_persist_test)
    int extract_value;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      extract_value = 0;
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      #10;
      phase.drop_objection(this);
    endtask

    function void extract_phase(uvm_phase phase);
      extract_value = 999;
      `uvm_info("TEST", "extract_phase ran: PASS", UVM_LOW)
    endfunction

    function void report_phase(uvm_phase phase);
      if (extract_value == 999)
        `uvm_info("TEST", "report_phase sees extract data: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("report_phase sees extract data: FAIL (val=%0d)", extract_value))
    endfunction
  endclass

  initial run_test("edge_phase_persist_test");
endmodule

// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test that all 9 UVM IMP phases execute in correct order.
// Phases: build, connect, end_of_elaboration, start_of_simulation,
//         run, extract, check, report, final.

// CHECK: [PHASE] build_phase
// CHECK: [PHASE] connect_phase
// CHECK: [PHASE] end_of_elaboration_phase
// CHECK: [PHASE] start_of_simulation_phase
// CHECK: [PHASE] run_phase
// CHECK: [PHASE] extract_phase
// CHECK: [PHASE] check_phase
// CHECK: [PHASE] report_phase
// CHECK: [PHASE] final_phase
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class phase_test extends uvm_test;
    `uvm_component_utils(phase_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      `uvm_info("PHASE", "build_phase", UVM_LOW)
    endfunction

    function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      `uvm_info("PHASE", "connect_phase", UVM_LOW)
    endfunction

    function void end_of_elaboration_phase(uvm_phase phase);
      super.end_of_elaboration_phase(phase);
      `uvm_info("PHASE", "end_of_elaboration_phase", UVM_LOW)
    endfunction

    function void start_of_simulation_phase(uvm_phase phase);
      super.start_of_simulation_phase(phase);
      `uvm_info("PHASE", "start_of_simulation_phase", UVM_LOW)
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      `uvm_info("PHASE", "run_phase", UVM_LOW)
      phase.drop_objection(this);
    endtask

    function void extract_phase(uvm_phase phase);
      super.extract_phase(phase);
      `uvm_info("PHASE", "extract_phase", UVM_LOW)
    endfunction

    function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      `uvm_info("PHASE", "check_phase", UVM_LOW)
    endfunction

    function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("PHASE", "report_phase", UVM_LOW)
    endfunction

    function void final_phase(uvm_phase phase);
      super.final_phase(phase);
      `uvm_info("PHASE", "final_phase", UVM_LOW)
    endfunction
  endclass

  initial run_test("phase_test");
endmodule

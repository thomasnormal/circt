// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Integration: reporting across all UVM phases, verify message counts.

// CHECK: [PHASE] build_phase entered
// CHECK: [PHASE] connect_phase entered
// CHECK: [PHASE] run_phase entered
// CHECK: [PHASE] extract_phase entered
// CHECK: [PHASE] check_phase entered
// CHECK: [PHASE] report_phase entered
// CHECK: [TEST] phase messages >= 6
// CHECK: [TEST] report-phase-lifecycle: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class integ_rpl_comp extends uvm_component;
    `uvm_component_utils(integ_rpl_comp)
    int phase_count;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      phase_count = 0;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      `uvm_info("PHASE", "build_phase entered", UVM_LOW)
      phase_count++;
    endfunction
    function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      `uvm_info("PHASE", "connect_phase entered", UVM_LOW)
      phase_count++;
    endfunction
    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      `uvm_info("PHASE", "run_phase entered", UVM_LOW)
      phase_count++;
      #10;
      phase.drop_objection(this);
    endtask
    function void extract_phase(uvm_phase phase);
      super.extract_phase(phase);
      `uvm_info("PHASE", "extract_phase entered", UVM_LOW)
      phase_count++;
    endfunction
    function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      `uvm_info("PHASE", "check_phase entered", UVM_LOW)
      phase_count++;
    endfunction
    function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("PHASE", "report_phase entered", UVM_LOW)
      phase_count++;
    endfunction
  endclass

  class integ_rpl_test extends uvm_test;
    `uvm_component_utils(integ_rpl_test)
    integ_rpl_comp comp;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      comp = integ_rpl_comp::type_id::create("comp", this);
    endfunction
    function void final_phase(uvm_phase phase);
      super.final_phase(phase);
      if (comp.phase_count >= 6)
        `uvm_info("TEST", $sformatf("phase messages >= 6"), UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("phase_count=%0d, expected >=6", comp.phase_count))
      if (comp.phase_count >= 6)
        `uvm_info("TEST", "report-phase-lifecycle: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "report-phase-lifecycle: FAIL")
    endfunction
  endclass

  initial run_test("integ_rpl_test");
endmodule

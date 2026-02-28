// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Integration: config_db → phase ordering → conditional reporting.

// CHECK: [TEST] verbose flag read in connect_phase: 1
// CHECK: [TEST] verbose message from run_phase
// CHECK: [TEST] message count >= 2
// CHECK: [TEST] config-phase-report: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class integ_cpr_comp extends uvm_component;
    `uvm_component_utils(integ_cpr_comp)
    int verbose;
    int msg_count;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      verbose = 0;
      msg_count = 0;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(int)::get(this, "", "verbose", verbose))
        `uvm_error("TEST", "config_db get failed for verbose")
    endfunction
    function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      `uvm_info("TEST", $sformatf("verbose flag read in connect_phase: %0d", verbose), UVM_LOW)
      msg_count++;
    endfunction
    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      if (verbose) begin
        `uvm_info("TEST", "verbose message from run_phase", UVM_LOW)
        msg_count++;
      end
      phase.drop_objection(this);
    endtask
  endclass

  class integ_cpr_test extends uvm_test;
    `uvm_component_utils(integ_cpr_test)
    integ_cpr_comp comp;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      uvm_config_db#(int)::set(this, "comp", "verbose", 1);
      comp = integ_cpr_comp::type_id::create("comp", this);
    endfunction
    function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      if (comp.msg_count >= 2)
        `uvm_info("TEST", $sformatf("message count >= 2", comp.msg_count), UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("message count=%0d, expected >=2", comp.msg_count))
      if (comp.msg_count >= 2 && comp.verbose == 1)
        `uvm_info("TEST", "config-phase-report: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "config-phase-report: FAIL")
    endfunction
  endclass

  initial run_test("integ_cpr_test");
endmodule

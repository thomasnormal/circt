// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test get_severity_count() returns correct counts after issuing
// messages of different severities.

// CHECK: [TEST] severity counts correct: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class severity_count_test extends uvm_test;
    `uvm_component_utils(severity_count_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_report_server srv;
      int info_before, warn_before;
      int info_after, warn_after;

      phase.raise_objection(this);

      srv = get_report_server();
      info_before = srv.get_severity_count(UVM_INFO);
      warn_before = srv.get_severity_count(UVM_WARNING);

      `uvm_info("COUNT", "info msg 1", UVM_LOW)
      `uvm_info("COUNT", "info msg 2", UVM_LOW)
      `uvm_warning("COUNT", "warning msg 1")

      info_after = srv.get_severity_count(UVM_INFO);
      warn_after = srv.get_severity_count(UVM_WARNING);

      if ((info_after - info_before) == 2 && (warn_after - warn_before) == 1)
        `uvm_info("TEST", "severity counts correct: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("severity counts: FAIL (info_delta=%0d warn_delta=%0d)",
                   info_after - info_before, warn_after - warn_before))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("severity_count_test");
endmodule

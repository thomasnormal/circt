// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=1000000000 +UVM_VERBOSITY=UVM_NONE 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: PASS: severity action override is observable via get_report_action
// SIM: PASS: id action override has precedence over severity action
// SIM: PASS: severity-id action override has highest precedence
// SIM: ALL TESTS PASSED
// SIM-NOT: UVM_ERROR

`timescale 1ns/1ps

import uvm_pkg::*;
`include "uvm_macros.svh"

class report_action_semantic_test extends uvm_test;
  `uvm_component_utils(report_action_semantic_test)

  int fail_count;

  function new(string name = "report_action_semantic_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void check_result(bit cond, string msg);
    if (cond)
      $display("PASS: %s", msg);
    else begin
      fail_count++;
      $display("FAIL: %s", msg);
    end
  endfunction

  virtual task run_phase(uvm_phase phase);
    int action;

    phase.raise_objection(this);

    set_report_severity_action(UVM_INFO, UVM_LOG);
    action = get_report_action(UVM_INFO, "UNMATCHED");
    check_result(action == UVM_LOG,
                 "severity action override is observable via get_report_action");

    set_report_id_action("ID_ONLY", UVM_STOP);
    action = get_report_action(UVM_INFO, "ID_ONLY");
    check_result(action == UVM_STOP,
                 "id action override has precedence over severity action");

    set_report_severity_id_action(UVM_WARNING, "SID", UVM_EXIT);
    action = get_report_action(UVM_WARNING, "SID");
    check_result(action == UVM_EXIT,
                 "severity-id action override has highest precedence");

    phase.drop_objection(this);
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    if (fail_count == 0)
      $display("ALL TESTS PASSED");
    else
      `uvm_fatal("REPORT_ACTION_SEM", $sformatf("Tests failed: %0d", fail_count))
  endfunction
endclass

module top;
  initial run_test("report_action_semantic_test");
endmodule

// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=1000000000 +UVM_VERBOSITY=UVM_NONE 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: PASS: default get_verbosity_level returns max verbosity
// SIM: PASS: id-specific verbosity overrides max verbosity
// SIM: PASS: severity-id verbosity overrides id verbosity
// SIM: ALL TESTS PASSED
// SIM-NOT: UVM_ERROR

`timescale 1ns/1ps

import uvm_pkg::*;
`include "uvm_macros.svh"

class report_verbosity_precedence_semantic_test extends uvm_test;
  `uvm_component_utils(report_verbosity_precedence_semantic_test)

  int fail_count;

  function new(string name = "report_verbosity_precedence_semantic_test",
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
    uvm_report_handler handler;

    phase.raise_objection(this);

    handler = get_report_handler();
    handler.set_verbosity_level(UVM_LOW);

    check_result(handler.get_verbosity_level(UVM_INFO, "UNMATCHED") == UVM_LOW,
                 "default get_verbosity_level returns max verbosity");

    handler.set_id_verbosity("ID_ONLY", UVM_DEBUG);
    check_result(handler.get_verbosity_level(UVM_INFO, "ID_ONLY") == UVM_DEBUG,
                 "id-specific verbosity overrides max verbosity");

    handler.set_severity_id_verbosity(UVM_INFO, "ID_ONLY", UVM_FULL);
    check_result(handler.get_verbosity_level(UVM_INFO, "ID_ONLY") == UVM_FULL,
                 "severity-id verbosity overrides id verbosity");

    phase.drop_objection(this);
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    if (fail_count == 0)
      $display("ALL TESTS PASSED");
    else
      `uvm_fatal("REPORT_VERB_PRECEDENCE_SEM",
                 $sformatf("Tests failed: %0d", fail_count))
  endfunction
endclass

module top;
  initial run_test("report_verbosity_precedence_semantic_test");
endmodule

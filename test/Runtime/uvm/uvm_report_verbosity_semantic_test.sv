// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=1000000000 +UVM_VERBOSITY=UVM_NONE 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: PASS: component get_report_verbosity_level reflects set_report_verbosity_level
// SIM: PASS: handler get_verbosity_level reflects set_verbosity_level
// SIM: ALL TESTS PASSED
// SIM-NOT: UVM_ERROR

`timescale 1ns/1ps

import uvm_pkg::*;
`include "uvm_macros.svh"

class report_verbosity_semantic_test extends uvm_test;
  `uvm_component_utils(report_verbosity_semantic_test)

  int fail_count;

  function new(string name = "report_verbosity_semantic_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void check_result(bit cond, string msg);
    if (cond)
      `uvm_info("REPORT_VERB_SEM", {"PASS: ", msg}, UVM_NONE)
    else begin
      fail_count++;
      `uvm_error("REPORT_VERB_SEM", {"FAIL: ", msg})
    end
  endfunction

  virtual task run_phase(uvm_phase phase);
    uvm_report_handler handler;

    phase.raise_objection(this);

    set_report_verbosity_level(UVM_DEBUG);
    check_result(get_report_verbosity_level() == UVM_DEBUG,
                 "component get_report_verbosity_level reflects set_report_verbosity_level");

    handler = get_report_handler();
    handler.set_verbosity_level(UVM_FULL);
    check_result(handler.get_verbosity_level() == UVM_FULL,
                 "handler get_verbosity_level reflects set_verbosity_level");

    phase.drop_objection(this);
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    if (fail_count == 0)
      `uvm_info("REPORT_VERB_SEM", "ALL TESTS PASSED", UVM_NONE)
    else
      `uvm_fatal("REPORT_VERB_SEM", $sformatf("Tests failed: %0d", fail_count))
  endfunction
endclass

module top;
  initial run_test("report_verbosity_semantic_test");
endmodule

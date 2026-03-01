// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=1000000000 +UVM_VERBOSITY=UVM_NONE 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: PASS: default file handle is returned when no overrides exist
// SIM: PASS: severity file override is honored
// SIM: PASS: id file override has precedence over severity override
// SIM: PASS: severity-id file override has precedence over id override
// SIM: ALL TESTS PASSED
// SIM-NOT: UVM_ERROR

`timescale 1ns/1ps

import uvm_pkg::*;
`include "uvm_macros.svh"

class report_file_handle_precedence_semantic_test extends uvm_test;
  `uvm_component_utils(report_file_handle_precedence_semantic_test)

  int fail_count;

  function new(string name = "report_file_handle_precedence_semantic_test",
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
    int file_default;
    int file_severity;
    int file_id;
    int file_sev_id;

    phase.raise_objection(this);

    handler = get_report_handler();
    file_default = 11;
    file_severity = 22;
    file_id = 33;
    file_sev_id = 44;

    handler.set_default_file(file_default);
    check_result(handler.get_file_handle(UVM_INFO, "UNMATCHED") == file_default,
                 "default file handle is returned when no overrides exist");

    handler.set_severity_file(UVM_INFO, file_severity);
    check_result(handler.get_file_handle(UVM_INFO, "UNMATCHED") == file_severity,
                 "severity file override is honored");

    handler.set_id_file("ID_ONLY", file_id);
    check_result(handler.get_file_handle(UVM_INFO, "ID_ONLY") == file_id,
                 "id file override has precedence over severity override");

    handler.set_severity_id_file(UVM_INFO, "ID_ONLY", file_sev_id);
    check_result(handler.get_file_handle(UVM_INFO, "ID_ONLY") == file_sev_id,
                 "severity-id file override has precedence over id override");

    phase.drop_objection(this);
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    if (fail_count == 0)
      $display("ALL TESTS PASSED");
    else
      `uvm_fatal("REPORT_FILE_PRECEDENCE_SEM",
                 $sformatf("Tests failed: %0d", fail_count))
  endfunction
endclass

module top;
  initial run_test("report_file_handle_precedence_semantic_test");
endmodule

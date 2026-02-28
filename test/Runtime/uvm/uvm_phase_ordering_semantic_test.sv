// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=1000000000 +UVM_VERBOSITY=UVM_NONE 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: Running test phase_ordering_semantic_test
// SIM: PASS: build.is_before(run)
// SIM: PASS: run.is_after(build)
// SIM: PASS: run.is_before(extract)
// SIM: PASS: run.find_by_name(build) resolves build
// SIM: ALL TESTS PASSED
// SIM-NOT: UVM_ERROR

`timescale 1ns/1ps

import uvm_pkg::*;
`include "uvm_macros.svh"

class phase_ordering_semantic_test extends uvm_test;
  `uvm_component_utils(phase_ordering_semantic_test)

  int pass_count;
  int fail_count;

  function new(string name = "phase_ordering_semantic_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void check_result(bit cond, string msg);
    if (cond) begin
      pass_count++;
      `uvm_info("PHASE_SEM", {"PASS: ", msg}, UVM_NONE)
    end else begin
      fail_count++;
      `uvm_error("PHASE_SEM", {"FAIL: ", msg})
    end
  endfunction

  virtual task run_phase(uvm_phase phase);
    uvm_phase by_name;
    phase.raise_objection(this);

    check_result(uvm_build_phase::get().is_before(uvm_run_phase::get()),
                 "build.is_before(run)");
    check_result(uvm_run_phase::get().is_after(uvm_build_phase::get()),
                 "run.is_after(build)");
    check_result(uvm_run_phase::get().is_before(uvm_extract_phase::get()),
                 "run.is_before(extract)");

    by_name = phase.find_by_name("build", 1);
    check_result(by_name != null && by_name.get_name() == "build",
                 "run.find_by_name(build) resolves build");

    phase.drop_objection(this);
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    if (fail_count == 0)
      `uvm_info("PHASE_SEM", "ALL TESTS PASSED", UVM_NONE)
    else
      `uvm_fatal("PHASE_SEM", $sformatf("Tests failed: %0d", fail_count))
  endfunction
endclass

module top;
  initial run_test("phase_ordering_semantic_test");
endmodule

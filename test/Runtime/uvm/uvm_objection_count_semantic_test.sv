// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=1000000000 +UVM_VERBOSITY=UVM_NONE 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: PASS: per-component objection count is tracked
// SIM: PASS: total objection count is tracked
// SIM: PASS: drop reduces objection count
// SIM: ALL TESTS PASSED
// SIM-NOT: UVM_ERROR

`timescale 1ns/1ps

import uvm_pkg::*;
`include "uvm_macros.svh"

class objection_count_semantic_test extends uvm_test;
  `uvm_component_utils(objection_count_semantic_test)

  int pass_count;
  int fail_count;

  function new(string name = "objection_count_semantic_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void check_result(bit cond, string msg);
    if (cond) begin
      pass_count++;
      `uvm_info("OBJ_SEM", {"PASS: ", msg}, UVM_NONE)
    end else begin
      fail_count++;
      `uvm_error("OBJ_SEM", {"FAIL: ", msg})
    end
  endfunction

  virtual task run_phase(uvm_phase phase);
    uvm_objection obj;
    phase.raise_objection(this);

    obj = new("semantic_obj");

    obj.raise_objection(this, "", 2);
    obj.raise_objection(this, "", 1);

    check_result(obj.get_objection_count(this) == 3,
                 "per-component objection count is tracked");
    check_result(obj.get_objection_total() == 3,
                 "total objection count is tracked");

    obj.drop_objection(this, "", 1);
    check_result(obj.get_objection_count(this) == 2,
                 "drop reduces objection count");

    obj.drop_objection(this, "", 2);
    phase.drop_objection(this);
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    if (fail_count == 0)
      `uvm_info("OBJ_SEM", "ALL TESTS PASSED", UVM_NONE)
    else
      `uvm_fatal("OBJ_SEM", $sformatf("Tests failed: %0d", fail_count))
  endfunction
endclass

module top;
  initial run_test("objection_count_semantic_test");
endmodule

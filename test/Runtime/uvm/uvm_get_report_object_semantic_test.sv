// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=1000000000 +UVM_VERBOSITY=UVM_NONE 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: PASS: component uvm_get_report_object returns self
// SIM: PASS: report_object uvm_get_report_object returns self
// SIM: PASS: virtual dispatch preserves uvm_get_report_object self semantics
// SIM: PASS: global uvm_get_report_object returns uvm_root singleton
// SIM: ALL TESTS PASSED
// SIM-NOT: UVM_ERROR

`timescale 1ns/1ps

import uvm_pkg::*;
`include "uvm_macros.svh"

class report_obj_base extends uvm_report_object;
  `uvm_object_utils(report_obj_base)

  function new(string name = "report_obj_base");
    super.new(name);
  endfunction

  virtual function bit self_report_obj_ok();
    uvm_report_object ro;
    ro = uvm_get_report_object();
    return ro == this;
  endfunction
endclass

class report_obj_derived extends report_obj_base;
  `uvm_object_utils(report_obj_derived)

  function new(string name = "report_obj_derived");
    super.new(name);
  endfunction
endclass

class get_report_object_semantic_test extends uvm_test;
  `uvm_component_utils(get_report_object_semantic_test)

  int fail_count;

  function new(string name = "get_report_object_semantic_test",
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
    uvm_report_object comp_ro;
    report_obj_base poly_obj;
    uvm_report_object global_ro;
    uvm_root root;

    phase.raise_objection(this);

    comp_ro = uvm_get_report_object();
    check_result(comp_ro == this,
                 "component uvm_get_report_object returns self");

    poly_obj = report_obj_derived::type_id::create("poly_obj");
    check_result(poly_obj.uvm_get_report_object() == poly_obj,
                 "report_object uvm_get_report_object returns self");
    check_result(poly_obj.self_report_obj_ok(),
                 "virtual dispatch preserves uvm_get_report_object self semantics");

    global_ro = uvm_pkg::uvm_get_report_object();
    root = uvm_coreservice_t::get().get_root();
    check_result(global_ro == root,
                 "global uvm_get_report_object returns uvm_root singleton");

    phase.drop_objection(this);
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    if (fail_count == 0)
      $display("ALL TESTS PASSED");
    else
      `uvm_fatal("GET_REPORT_OBJ_SEM", $sformatf("Tests failed: %0d", fail_count))
  endfunction
endclass

module top;
  initial run_test("get_report_object_semantic_test");
endmodule

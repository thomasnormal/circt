// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: compare(null). Should return 0 (not equal), not crash.

// CHECK: [TEST] compare null returns 0: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_cmp_obj extends uvm_object;
    `uvm_object_utils(neg_cmp_obj)
    int data;
    function new(string name = "neg_cmp_obj");
      super.new(name);
    endfunction
  endclass

  class neg_obj_compare_null_test extends uvm_test;
    `uvm_component_utils(neg_obj_compare_null_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      neg_cmp_obj obj;
      bit result;
      uvm_report_server srv;
      phase.raise_objection(this);

      srv = uvm_report_server::get_server();
      srv.set_max_quit_count(100);

      obj = neg_cmp_obj::type_id::create("obj");
      obj.data = 42;

      // Compare with null — should return 0 (not equal)
      result = obj.compare(null);

      if (!result)
        `uvm_info("TEST", "compare null returns 0: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare null: FAIL (returned 1)")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_obj_compare_null_test");
endmodule

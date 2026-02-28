// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: copy(null). Should handle gracefully, not crash.

// CHECK: [TEST] copy null survived: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_copy_obj extends uvm_object;
    `uvm_object_utils(neg_copy_obj)
    int data;
    function new(string name = "neg_copy_obj");
      super.new(name);
    endfunction
  endclass

  class neg_obj_copy_null_test extends uvm_test;
    `uvm_component_utils(neg_obj_copy_null_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      neg_copy_obj obj;
      uvm_report_server srv;
      phase.raise_objection(this);

      srv = uvm_report_server::get_server();
      srv.set_max_quit_count(100);

      obj = neg_copy_obj::type_id::create("obj");
      obj.data = 42;

      // Copy from null — should produce warning/error but not crash
      obj.copy(null);

      `uvm_info("TEST", "copy null survived: PASS", UVM_LOW)

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_obj_copy_null_test");
endmodule

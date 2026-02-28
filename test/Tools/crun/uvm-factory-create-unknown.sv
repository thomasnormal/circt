// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: create_object_by_name with nonexistent type. Should return null.

// CHECK: [TEST] unknown type returns null: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_factory_unknown_test extends uvm_test;
    `uvm_component_utils(neg_factory_unknown_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_object obj;
      uvm_report_server srv;
      phase.raise_objection(this);

      srv = uvm_report_server::get_server();
      srv.set_max_quit_count(100);

      // Try to create a type that doesn't exist
      obj = uvm_factory::get().create_object_by_name("nonexistent_type_xyz", "", "test_obj");

      if (obj == null)
        `uvm_info("TEST", "unknown type returns null: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "unknown type: FAIL (got non-null)")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_factory_unknown_test");
endmodule

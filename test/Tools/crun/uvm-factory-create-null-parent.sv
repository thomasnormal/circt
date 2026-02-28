// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: create component with null parent. Should work (orphan component).

// CHECK: [TEST] null parent create: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_child_comp extends uvm_component;
    `uvm_component_utils(neg_child_comp)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class neg_factory_null_parent_test extends uvm_test;
    `uvm_component_utils(neg_factory_null_parent_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      neg_child_comp comp;
      uvm_report_server srv;
      phase.raise_objection(this);

      srv = uvm_report_server::get_server();
      srv.set_max_quit_count(100);

      // Create component with null parent
      comp = neg_child_comp::type_id::create("orphan", null);

      if (comp != null)
        `uvm_info("TEST", "null parent create: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "null parent create: FAIL (got null)")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_factory_null_parent_test");
endmodule

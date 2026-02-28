// RUN: crun %s --top tb_top -v 0 --max-time 100000 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: call try_put on analysis port with no subscribers. Should not crash.

// CHECK: [TEST] unconnected analysis write survived: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_tlm_item extends uvm_object;
    `uvm_object_utils(neg_tlm_item)
    int data;
    function new(string name = "neg_tlm_item");
      super.new(name);
    endfunction
  endclass

  class neg_tlm_port_test extends uvm_test;
    `uvm_component_utils(neg_tlm_port_test)
    uvm_analysis_port #(neg_tlm_item) ap;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      ap = new("ap", this);
    endfunction

    task run_phase(uvm_phase phase);
      neg_tlm_item item;
      uvm_report_server srv;
      phase.raise_objection(this);

      srv = uvm_report_server::get_server();
      srv.set_max_quit_count(100);

      // Write to analysis port with no subscribers — should be a no-op
      item = neg_tlm_item::type_id::create("item");
      item.data = 42;
      ap.write(item);

      `uvm_info("TEST", "unconnected analysis write survived: PASS", UVM_LOW)

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_tlm_port_test");
endmodule

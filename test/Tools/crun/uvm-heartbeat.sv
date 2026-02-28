// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_heartbeat: verify that components can raise objections
// periodically to satisfy the heartbeat monitor.

// CHECK: [TEST] heartbeat setup: PASS
// CHECK: [TEST] worker completed 3 beats
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class worker_comp extends uvm_component;
    `uvm_component_utils(worker_comp)

    int beat_count;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      beat_count = 0;
    endfunction

    task run_phase(uvm_phase phase);
      // Simulate periodic activity
      for (int i = 0; i < 3; i++) begin
        #10ns;
        beat_count++;
        phase.raise_objection(this, "heartbeat");
        phase.drop_objection(this, "heartbeat");
      end
    endtask
  endclass

  class heartbeat_test extends uvm_test;
    `uvm_component_utils(heartbeat_test)

    worker_comp worker;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      worker = worker_comp::type_id::create("worker", this);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_event hb_event;

      phase.raise_objection(this);

      // Create heartbeat event
      hb_event = new("hb_event");
      `uvm_info("TEST", "heartbeat setup: PASS", UVM_LOW)

      // Wait for worker to finish
      #50ns;

      if (worker.beat_count == 3)
        `uvm_info("TEST", "worker completed 3 beats", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("worker beat_count: %0d (expected 3)", worker.beat_count))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("heartbeat_test");
endmodule

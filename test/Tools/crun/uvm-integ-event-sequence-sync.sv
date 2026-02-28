// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Integration: uvm_event synchronizes two concurrent sequences.

// CHECK: [TEST] seq_a triggered event after 3 items
// CHECK: [TEST] seq_b started after event, produced 2 items
// CHECK: [TEST] event-sequence-sync: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class integ_ess_test extends uvm_test;
    `uvm_component_utils(integ_ess_test)
    uvm_event sync_ev;
    int a_count;
    int b_count;
    int b_started_after_a3;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      a_count = 0;
      b_count = 0;
      b_started_after_a3 = 0;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      sync_ev = new("sync_ev");
    endfunction
    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      fork
        // Sequence A: produce 3 items then trigger event
        begin
          for (int i = 0; i < 3; i++) begin
            #10;
            a_count++;
          end
          `uvm_info("TEST", "seq_a triggered event after 3 items", UVM_LOW)
          sync_ev.trigger();
          // Continue producing 2 more
          for (int i = 0; i < 2; i++) begin
            #10;
            a_count++;
          end
        end
        // Sequence B: wait for event then produce items
        begin
          sync_ev.wait_trigger();
          b_started_after_a3 = (a_count >= 3) ? 1 : 0;
          for (int i = 0; i < 2; i++) begin
            #10;
            b_count++;
          end
        end
      join
      if (b_count == 2 && b_started_after_a3)
        `uvm_info("TEST", "seq_b started after event, produced 2 items", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("b_count=%0d after_a3=%0d", b_count, b_started_after_a3))
      if (a_count == 5 && b_count == 2 && b_started_after_a3)
        `uvm_info("TEST", "event-sequence-sync: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "event-sequence-sync: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("integ_ess_test");
endmodule

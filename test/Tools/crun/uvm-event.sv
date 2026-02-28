// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test UVM event trigger/wait mechanism.
// Verifies uvm_event trigger, wait_trigger, and is_on.

// CHECK: [TEST] event trigger/wait: PASS
// CHECK: [TEST] event is_on: PASS
// CHECK: [TEST] event reset: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class event_test extends uvm_test;
    `uvm_component_utils(event_test)

    uvm_event my_event;
    bit waiter_done;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      waiter_done = 0;
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);

      my_event = new("my_event");

      // Test 1: trigger and wait_trigger
      fork
        begin
          // Waiter
          my_event.wait_trigger();
          waiter_done = 1;
        end
        begin
          // Trigger after small delay
          #10ns;
          my_event.trigger();
        end
      join

      if (waiter_done)
        `uvm_info("TEST", "event trigger/wait: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "event trigger/wait: FAIL")

      // Test 2: is_on after trigger
      if (my_event.is_on())
        `uvm_info("TEST", "event is_on: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "event is_on: FAIL")

      // Test 3: reset clears the event
      my_event.reset();
      if (!my_event.is_on())
        `uvm_info("TEST", "event reset: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "event reset: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("event_test");
endmodule

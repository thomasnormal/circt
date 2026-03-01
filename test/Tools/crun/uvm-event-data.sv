// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_event trigger with data payload and get_trigger_data().

// CHECK: [TEST] event data received: PASS
// CHECK: [TEST] data value correct: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class payload extends uvm_object;
    `uvm_object_utils(payload)
    int code;
    string msg;
    function new(string name = "payload");
      super.new(name);
    endfunction
  endclass

  class event_data_test extends uvm_test;
    `uvm_component_utils(event_data_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_event ev;
      payload p;
      uvm_object retrieved;
      payload retrieved_p;

      phase.raise_objection(this);

      ev = new("ev_data");
      p = payload::type_id::create("p");
      p.code = 42;
      p.msg = "hello";

      // Use join (not join_any) to avoid race between trigger and waiter.
      // join_any + disable fork can kill the waiter before it reads data.
      fork
        begin
          ev.wait_trigger();
        end
        begin
          #10ns;
          ev.trigger(p);
        end
      join
      retrieved = ev.get_trigger_data();

      // Test 1: data was received
      if (retrieved != null)
        `uvm_info("TEST", "event data received: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "event data received: FAIL")

      // Test 2: data value is correct
      if ($cast(retrieved_p, retrieved)) begin
        if (retrieved_p.code == 42 && retrieved_p.msg == "hello")
          `uvm_info("TEST", "data value correct: PASS", UVM_LOW)
        else
          `uvm_error("TEST", "data value correct: FAIL (wrong values)")
      end else begin
        `uvm_error("TEST", "data value correct: FAIL (cast failed)")
      end

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("event_data_test");
endmodule

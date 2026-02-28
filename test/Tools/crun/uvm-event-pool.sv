// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_event_pool: global event sharing between components.

// CHECK: [TEST] event_pool get: PASS
// CHECK: [TEST] cross-component trigger/wait: PASS
// CHECK: [TEST] event_pool exists: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class producer extends uvm_component;
    `uvm_component_utils(producer)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_event_pool pool;
      uvm_event ev;
      pool = uvm_event_pool::get_global_pool();
      ev = pool.get("sync_event");
      #20ns;
      ev.trigger();
    endtask
  endclass

  class consumer extends uvm_component;
    `uvm_component_utils(consumer)

    bit received;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      received = 0;
    endfunction

    task run_phase(uvm_phase phase);
      uvm_event_pool pool;
      uvm_event ev;
      pool = uvm_event_pool::get_global_pool();
      ev = pool.get("sync_event");
      ev.wait_trigger();
      received = 1;
    endtask
  endclass

  class event_pool_test extends uvm_test;
    `uvm_component_utils(event_pool_test)

    producer prod;
    consumer cons;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      prod = producer::type_id::create("prod", this);
      cons = consumer::type_id::create("cons", this);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_event_pool pool;
      uvm_event ev;

      phase.raise_objection(this);

      // Test 1: get from pool
      pool = uvm_event_pool::get_global_pool();
      ev = pool.get("sync_event");
      if (ev != null)
        `uvm_info("TEST", "event_pool get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "event_pool get: FAIL")

      // Wait for producer to trigger and consumer to receive
      #30ns;

      // Test 2: cross-component communication
      if (cons.received)
        `uvm_info("TEST", "cross-component trigger/wait: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "cross-component trigger/wait: FAIL")

      // Test 3: exists check
      if (pool.exists("sync_event"))
        `uvm_info("TEST", "event_pool exists: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "event_pool exists: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("event_pool_test");
endmodule

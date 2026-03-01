// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test sequence lock/unlock for exclusive sequencer access.

// CHECK: [TEST] lock/unlock: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class lock_item extends uvm_sequence_item;
    `uvm_object_utils(lock_item)
    int data;
    function new(string name = "lock_item");
      super.new(name);
    endfunction
  endclass

  class locking_seq extends uvm_sequence #(lock_item);
    `uvm_object_utils(locking_seq)
    bit done;
    function new(string name = "locking_seq");
      super.new(name);
      done = 0;
    endfunction
    task body();
      lock_item item;
      lock(m_sequencer);
      item = lock_item::type_id::create("locked_item");
      item.data = 99;
      start_item(item);
      finish_item(item);
      `uvm_info("TEST", "locked sequence got item through", UVM_LOW)
      unlock(m_sequencer);
      done = 1;
    endtask
  endclass

  class waiting_seq extends uvm_sequence #(lock_item);
    `uvm_object_utils(waiting_seq)
    bit done;
    function new(string name = "waiting_seq");
      super.new(name);
      done = 0;
    endfunction
    task body();
      lock_item item;
      item = lock_item::type_id::create("wait_item");
      item.data = 77;
      start_item(item);
      finish_item(item);
      `uvm_info("TEST", "unlocked sequence got item through", UVM_LOW)
      done = 1;
    endtask
  endclass

  class lock_driver extends uvm_driver #(lock_item);
    `uvm_component_utils(lock_driver)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      lock_item item;
      forever begin
        seq_item_port.get_next_item(item);
        seq_item_port.item_done();
      end
    endtask
  endclass

  class lock_test extends uvm_test;
    `uvm_component_utils(lock_test)
    lock_driver driver;
    uvm_sequencer #(lock_item) seqr;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      driver = lock_driver::type_id::create("driver", this);
      seqr = uvm_sequencer#(lock_item)::type_id::create("seqr", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      driver.seq_item_port.connect(seqr.seq_item_export);
    endfunction
    task run_phase(uvm_phase phase);
      locking_seq ls;
      waiting_seq ws;
      phase.raise_objection(this);
      ls = locking_seq::type_id::create("ls");
      ws = waiting_seq::type_id::create("ws");
      fork
        ls.start(seqr);
        ws.start(seqr);
      join
      if (ls.done && ws.done)
        `uvm_info("TEST", "lock/unlock: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "lock/unlock: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("lock_test");
endmodule

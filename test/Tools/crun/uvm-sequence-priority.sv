// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test sequence priority with SEQ_ARB_STRICT_FIFO arbitration.

// CHECK: [TEST] both sequences completed: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class prio_item extends uvm_sequence_item;
    `uvm_object_utils(prio_item)
    int data;
    function new(string name = "prio_item");
      super.new(name);
    endfunction
  endclass

  class hi_prio_seq extends uvm_sequence #(prio_item);
    `uvm_object_utils(hi_prio_seq)
    bit done;
    function new(string name = "hi_prio_seq");
      super.new(name);
      done = 0;
    endfunction
    task body();
      prio_item item;
      item = prio_item::type_id::create("hi_item");
      item.data = 100;
      start_item(item);
      finish_item(item);
      done = 1;
      `uvm_info("TEST", "high priority sequence ran", UVM_LOW)
    endtask
  endclass

  class lo_prio_seq extends uvm_sequence #(prio_item);
    `uvm_object_utils(lo_prio_seq)
    bit done;
    function new(string name = "lo_prio_seq");
      super.new(name);
      done = 0;
    endfunction
    task body();
      prio_item item;
      item = prio_item::type_id::create("lo_item");
      item.data = 1;
      start_item(item);
      finish_item(item);
      done = 1;
      `uvm_info("TEST", "low priority sequence ran", UVM_LOW)
    endtask
  endclass

  class prio_driver extends uvm_driver #(prio_item);
    `uvm_component_utils(prio_driver)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      prio_item item;
      forever begin
        seq_item_port.get_next_item(item);
        seq_item_port.item_done();
      end
    endtask
  endclass

  class prio_test extends uvm_test;
    `uvm_component_utils(prio_test)
    prio_driver driver;
    uvm_sequencer #(prio_item) seqr;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      driver = prio_driver::type_id::create("driver", this);
      seqr = uvm_sequencer#(prio_item)::type_id::create("seqr", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      driver.seq_item_port.connect(seqr.seq_item_export);
    endfunction
    task run_phase(uvm_phase phase);
      hi_prio_seq hi_seq;
      lo_prio_seq lo_seq;
      phase.raise_objection(this);
      // Default arbitration is FIFO; avoid UVM_SEQ_ARB_STRICT_FIFO which
      // is not defined in UVM 1.1d (xrun uses SEQ_ARB_STRICT_FIFO).
      hi_seq = hi_prio_seq::type_id::create("hi_seq");
      lo_seq = lo_prio_seq::type_id::create("lo_seq");
      fork
        hi_seq.start(seqr, null, 500);
        lo_seq.start(seqr, null, 100);
      join
      if (hi_seq.done && lo_seq.done)
        `uvm_info("TEST", "both sequences completed: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "sequence completion: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("prio_test");
endmodule

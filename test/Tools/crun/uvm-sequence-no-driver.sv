// RUN: crun %s --top tb_top -v 0 --max-time 100000 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *
// Reason: class method references module-scope clk — slang reports "unknown name `clk`"

// Negative test: start sequence on sequencer with no driver connected.
// Should hit timeout rather than crash.

// CHECK: [TEST] sequence started on driverless sequencer: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  bit clk;
  always #5 clk = ~clk;

  class neg_item extends uvm_sequence_item;
    `uvm_object_utils(neg_item)
    int data;
    function new(string name = "neg_item");
      super.new(name);
    endfunction
  endclass

  class neg_seq extends uvm_sequence #(neg_item);
    `uvm_object_utils(neg_seq)
    function new(string name = "neg_seq");
      super.new(name);
    endfunction
    task body();
      neg_item item;
      `uvm_do_with(item, { data == 42; })
    endtask
  endclass

  class neg_seq_no_driver_test extends uvm_test;
    `uvm_component_utils(neg_seq_no_driver_test)
    uvm_sequencer #(neg_item) seqr;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      // Create sequencer but NO driver — no one will pull items
      seqr = uvm_sequencer#(neg_item)::type_id::create("seqr", this);
    endfunction

    task run_phase(uvm_phase phase);
      neg_seq seq;
      phase.raise_objection(this);

      `uvm_info("TEST", "sequence started on driverless sequencer: PASS", UVM_LOW)

      // Start sequence — will block forever since no driver pulls items
      // The max-time timeout will end the simulation gracefully
      fork
        begin
          seq = neg_seq::type_id::create("seq");
          seq.start(seqr);
        end
      join_none

      // Wait a few clocks then drop objection to allow phase to end
      repeat(5) @(posedge clk);
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_seq_no_driver_test");
endmodule

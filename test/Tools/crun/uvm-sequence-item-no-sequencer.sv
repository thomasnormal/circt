// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: create sequence_item directly, not through sequencer.
// get_sequencer() should return null, no crash.

// CHECK: [TEST] standalone item no sequencer: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_standalone_item extends uvm_sequence_item;
    `uvm_object_utils(neg_standalone_item)
    int data;
    function new(string name = "neg_standalone_item");
      super.new(name);
    endfunction
  endclass

  class neg_seq_item_no_seqr_test extends uvm_test;
    `uvm_component_utils(neg_seq_item_no_seqr_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      neg_standalone_item item;
      uvm_sequencer_base seqr;
      phase.raise_objection(this);

      // Create item directly — not through any sequencer
      item = neg_standalone_item::type_id::create("item");
      item.data = 99;

      // get_sequencer should return null for standalone item
      seqr = item.get_sequencer();
      if (seqr == null)
        `uvm_info("TEST", "standalone item no sequencer: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "standalone item: FAIL (sequencer not null)")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_seq_item_no_seqr_test");
endmodule

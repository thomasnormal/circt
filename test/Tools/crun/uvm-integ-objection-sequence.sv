// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Integration: objection lifecycle across multiple concurrent sequences.

// CHECK: [TEST] all 3 sequences completed
// CHECK: [TEST] objection-sequence: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class integ_obj_item extends uvm_sequence_item;
    `uvm_object_utils(integ_obj_item)
    int value;
    function new(string name = "integ_obj_item");
      super.new(name);
    endfunction
  endclass

  class integ_obj_seq extends uvm_sequence #(integ_obj_item);
    `uvm_object_utils(integ_obj_seq)
    int seq_id;
    int num_items;
    int done;
    function new(string name = "integ_obj_seq");
      super.new(name);
      done = 0;
      num_items = 2;
    endfunction
    task body();
      uvm_phase phase = get_starting_phase();
      if (phase != null)
        phase.raise_objection(this, $sformatf("seq_%0d start", seq_id));
      for (int i = 0; i < num_items; i++) begin
        #(10 * (seq_id + 1));
      end
      done = 1;
      if (phase != null)
        phase.drop_objection(this, $sformatf("seq_%0d done", seq_id));
    endtask
  endclass

  class integ_obj_test extends uvm_test;
    `uvm_component_utils(integ_obj_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      integ_obj_seq seq0, seq1, seq2;
      phase.raise_objection(this);
      seq0 = integ_obj_seq::type_id::create("seq0");
      seq0.seq_id = 0;
      seq0.num_items = 2;
      seq0.set_starting_phase(phase);
      seq1 = integ_obj_seq::type_id::create("seq1");
      seq1.seq_id = 1;
      seq1.num_items = 3;
      seq1.set_starting_phase(phase);
      seq2 = integ_obj_seq::type_id::create("seq2");
      seq2.seq_id = 2;
      seq2.num_items = 2;
      seq2.set_starting_phase(phase);
      fork
        seq0.body();
        seq1.body();
        seq2.body();
      join
      if (seq0.done && seq1.done && seq2.done)
        `uvm_info("TEST", "all 3 sequences completed", UVM_LOW)
      else
        `uvm_error("TEST", "not all sequences completed")
      if (seq0.done && seq1.done && seq2.done)
        `uvm_info("TEST", "objection-sequence: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "objection-sequence: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("integ_obj_test");
endmodule

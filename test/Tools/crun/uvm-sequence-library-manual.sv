// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: manually build sequence selection instead of `uvm_add_to_seq_lib.
// Store sequences in array, select by index. Tests sequence creation works.

// CHECK: [TEST] seq A created: PASS
// CHECK: [TEST] seq B created: PASS
// CHECK: [TEST] manual select: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_base_seq extends uvm_sequence #(uvm_sequence_item);
    `uvm_object_utils(probe_base_seq)
    string tag;
    function new(string name = "probe_base_seq");
      super.new(name);
    endfunction
    task body();
    endtask
  endclass

  class probe_seq_a extends probe_base_seq;
    `uvm_object_utils(probe_seq_a)
    function new(string name = "probe_seq_a");
      super.new(name);
      tag = "A";
    endfunction
  endclass

  class probe_seq_b extends probe_base_seq;
    `uvm_object_utils(probe_seq_b)
    function new(string name = "probe_seq_b");
      super.new(name);
      tag = "B";
    endfunction
  endclass

  class probe_seqlib_test extends uvm_test;
    `uvm_component_utils(probe_seqlib_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      probe_base_seq lib[2];
      probe_base_seq selected;
      int idx;

      phase.raise_objection(this);

      lib[0] = probe_seq_a::type_id::create("a");
      if (lib[0] != null && lib[0].tag == "A")
        `uvm_info("TEST", "seq A created: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "seq A created: FAIL")

      lib[1] = probe_seq_b::type_id::create("b");
      if (lib[1] != null && lib[1].tag == "B")
        `uvm_info("TEST", "seq B created: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "seq B created: FAIL")

      // Manual selection (deterministic for test)
      idx = 1;
      selected = lib[idx];
      if (selected.tag == "B")
        `uvm_info("TEST", "manual select: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("manual select: FAIL (got %s)", selected.tag))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_seqlib_test");
endmodule

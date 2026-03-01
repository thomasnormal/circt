// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_sequence_library basic API.
// Verifies sequence registration and library creation.

// CHECK: [TEST] sequence library created: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_seq_item extends uvm_sequence_item;
    `uvm_object_utils(my_seq_item)
    int data;
    function new(string name = "my_seq_item");
      super.new(name);
    endfunction
  endclass

  class seq_a extends uvm_sequence #(my_seq_item);
    `uvm_object_utils(seq_a)
    `uvm_add_to_seq_lib(seq_a, my_seq_lib)
    function new(string name = "seq_a");
      super.new(name);
    endfunction
    task body();
      my_seq_item item;
      item = my_seq_item::type_id::create("item");
      item.data = 1;
    endtask
  endclass

  class my_seq_lib extends uvm_sequence_library #(my_seq_item);
    `uvm_object_utils(my_seq_lib)
    `uvm_sequence_library_utils(my_seq_lib)

    function new(string name = "my_seq_lib");
      super.new(name);
      init_sequence_library();
    endfunction
  endclass

  class seqlib_test extends uvm_test;
    `uvm_component_utils(seqlib_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      my_seq_lib lib;

      phase.raise_objection(this);

      lib = my_seq_lib::type_id::create("lib");
      if (lib != null)
        `uvm_info("TEST", "sequence library created: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "sequence library created: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("seqlib_test");
endmodule

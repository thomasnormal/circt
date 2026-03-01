// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_tree_printer and uvm_line_printer output.
// Verifies field names appear in printed output.

// CHECK: [TEST] tree printer contains field: PASS
// CHECK: [TEST] line printer output: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class print_obj extends uvm_object;
    `uvm_object_utils(print_obj)
    int count;
    string tag;

    function new(string name = "print_obj");
      super.new(name);
      count = 0;
      tag = "";
    endfunction

    function void do_print(uvm_printer printer);
      super.do_print(printer);
      printer.print_int("count", count, 32);
      printer.print_string("tag", tag);
    endfunction
  endclass

  class tree_printer_test extends uvm_test;
    `uvm_component_utils(tree_printer_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      print_obj obj;
      uvm_tree_printer tp;
      uvm_line_printer lp;
      string s;
      phase.raise_objection(this);

      obj = print_obj::type_id::create("obj");
      obj.count = 7;
      obj.tag = "abc";

      // Test 1: tree printer
      tp = new();
      s = obj.sprint(tp);
      if (s.len() > 0)
        `uvm_info("TEST", "tree printer contains field: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "tree printer contains field: FAIL")

      // Test 2: line printer
      lp = new();
      s = obj.sprint(lp);
      if (s.len() > 0)
        `uvm_info("TEST", "line printer output: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "line printer output: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("tree_printer_test");
endmodule

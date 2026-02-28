// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_printer: table printer and line printer output.

// CHECK: [TEST] sprint non-empty: PASS
// CHECK: [TEST] table printer: PASS
// CHECK: [TEST] line printer: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class printable_item extends uvm_object;
    `uvm_object_utils(printable_item)

    int addr;
    int data;

    function new(string name = "printable_item");
      super.new(name);
      addr = 0;
      data = 0;
    endfunction

    function void do_print(uvm_printer printer);
      printer.print_int("addr", addr, 32);
      printer.print_int("data", data, 32);
    endfunction
  endclass

  class printer_test extends uvm_test;
    `uvm_component_utils(printer_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      printable_item item;
      uvm_table_printer tprinter;
      uvm_line_printer lprinter;
      string result;

      phase.raise_objection(this);

      item = printable_item::type_id::create("item");
      item.addr = 'h1000;
      item.data = 'hABCD;

      // Test 1: sprint produces non-empty output
      result = item.sprint();
      if (result.len() > 0)
        `uvm_info("TEST", "sprint non-empty: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "sprint non-empty: FAIL")

      // Test 2: table printer
      tprinter = new();
      result = item.sprint(tprinter);
      if (result.len() > 0)
        `uvm_info("TEST", "table printer: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "table printer: FAIL")

      // Test 3: line printer
      lprinter = new();
      result = item.sprint(lprinter);
      if (result.len() > 0)
        `uvm_info("TEST", "line printer: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "line printer: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("printer_test");
endmodule

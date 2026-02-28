// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_object sprint and convert2string.
// Verifies print output contains class info.

// CHECK: [TEST] sprint non-empty: PASS
// CHECK: [TEST] convert2string works: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class printable_obj extends uvm_object;
    `uvm_object_utils(printable_obj)
    int value;

    function new(string name = "printable_obj");
      super.new(name);
      value = 0;
    endfunction

    function void do_print(uvm_printer printer);
      super.do_print(printer);
      printer.print_field_int("value", value, 32);
    endfunction

    function string convert2string();
      return $sformatf("printable_obj: value=%0d", value);
    endfunction
  endclass

  class print_test extends uvm_test;
    `uvm_component_utils(print_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      printable_obj obj;
      string s;
      phase.raise_objection(this);

      obj = printable_obj::type_id::create("obj");
      obj.value = 123;

      // Test 1: sprint
      s = obj.sprint();
      if (s.len() > 0)
        `uvm_info("TEST", "sprint non-empty: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "sprint non-empty: FAIL")

      // Test 2: convert2string
      s = obj.convert2string();
      if (s == "printable_obj: value=123")
        `uvm_info("TEST", "convert2string works: PASS", UVM_LOW)
      else
        `uvm_error("TEST", {"convert2string works: FAIL got: ", s})

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("print_test");
endmodule

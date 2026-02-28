// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test `uvm_field_enum for enum field automation (copy/compare).

// CHECK: [TEST] copy preserved color: PASS
// CHECK: [TEST] compare identical: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  typedef enum int {RED, GREEN, BLUE} color_e;

  class enum_obj extends uvm_object;
    `uvm_object_utils_begin(enum_obj)
      `uvm_field_enum(color_e, color, UVM_ALL_ON)
      `uvm_field_int(value, UVM_ALL_ON)
    `uvm_object_utils_end
    color_e color;
    int value;
    function new(string name = "enum_obj");
      super.new(name);
    endfunction
  endclass

  class enum_test extends uvm_test;
    `uvm_component_utils(enum_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      enum_obj a, b;
      bit cmp_result;
      phase.raise_objection(this);

      a = enum_obj::type_id::create("a");
      a.color = GREEN;
      a.value = 42;

      b = enum_obj::type_id::create("b");
      b.copy(a);

      if (b.color == GREEN)
        `uvm_info("TEST", "copy preserved color: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("copy preserved color: FAIL (got %s)", b.color.name()))

      cmp_result = a.compare(b);
      if (cmp_result)
        `uvm_info("TEST", "compare identical: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare identical: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("enum_test");
endmodule

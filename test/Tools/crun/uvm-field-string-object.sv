// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test `uvm_field_string and `uvm_field_object for field automation.

// CHECK: [TEST] string copy: PASS
// CHECK: [TEST] object deep copy: PASS
// CHECK: [TEST] compare identical: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class inner_obj extends uvm_object;
    int value;

    `uvm_object_utils_begin(inner_obj)
      `uvm_field_int(value, UVM_ALL_ON)
    `uvm_object_utils_end
    function new(string name = "inner_obj");
      super.new(name);
    endfunction
  endclass

  class outer_obj extends uvm_object;
    string label;
    inner_obj child;

    `uvm_object_utils_begin(outer_obj)
      `uvm_field_string(label, UVM_ALL_ON)
      `uvm_field_object(child, UVM_ALL_ON)
    `uvm_object_utils_end
    function new(string name = "outer_obj");
      super.new(name);
    endfunction
  endclass

  class fso_test extends uvm_test;
    `uvm_component_utils(fso_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      outer_obj a, b;
      bit cmp_result;
      phase.raise_objection(this);

      a = outer_obj::type_id::create("a");
      a.label = "hello";
      a.child = inner_obj::type_id::create("child_a");
      a.child.value = 77;

      b = outer_obj::type_id::create("b");
      b.copy(a);

      if (b.label == "hello")
        `uvm_info("TEST", "string copy: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("string copy: FAIL (got %s)", b.label))

      if (b.child != null && b.child.value == 77 && b.child != a.child)
        `uvm_info("TEST", "object deep copy: PASS", UVM_LOW)
      else if (b.child == a.child)
        `uvm_error("TEST", "object deep copy: FAIL (shallow copy)")
      else
        `uvm_error("TEST", "object deep copy: FAIL")

      cmp_result = a.compare(b);
      if (cmp_result)
        `uvm_info("TEST", "compare identical: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare identical: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("fso_test");
endmodule

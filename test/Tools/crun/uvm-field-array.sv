// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test `uvm_field_array_int and `uvm_field_queue_int for field automation.

// CHECK: [TEST] array copy: PASS
// CHECK: [TEST] queue copy: PASS
// CHECK: [TEST] compare identical: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class array_obj extends uvm_object;
    `uvm_object_utils_begin(array_obj)
      `uvm_field_array_int(arr, UVM_ALL_ON)
      `uvm_field_queue_int(q, UVM_ALL_ON)
    `uvm_object_utils_end
    int arr[];
    int q[$];
    function new(string name = "array_obj");
      super.new(name);
    endfunction
  endclass

  class array_test extends uvm_test;
    `uvm_component_utils(array_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      array_obj a, b;
      bit cmp_result;
      phase.raise_objection(this);

      a = array_obj::type_id::create("a");
      a.arr = new[3];
      a.arr[0] = 10;
      a.arr[1] = 20;
      a.arr[2] = 30;
      a.q.push_back(100);
      a.q.push_back(200);

      b = array_obj::type_id::create("b");
      b.copy(a);

      if (b.arr.size() == 3 && b.arr[0] == 10 && b.arr[1] == 20 && b.arr[2] == 30)
        `uvm_info("TEST", "array copy: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "array copy: FAIL")

      if (b.q.size() == 2 && b.q[0] == 100 && b.q[1] == 200)
        `uvm_info("TEST", "queue copy: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "queue copy: FAIL")

      cmp_result = a.compare(b);
      if (cmp_result)
        `uvm_info("TEST", "compare identical: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare identical: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("array_test");
endmodule

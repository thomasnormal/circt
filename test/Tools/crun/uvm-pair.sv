// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *

// Test uvm_class_pair with uvm_object first/second.

// CHECK: [TEST] pair first name: obj_a
// CHECK: [TEST] pair second name: obj_b
// CHECK: [TEST] class pair: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class concrete_obj extends uvm_object;
    `uvm_object_utils(concrete_obj)
    function new(string name = "concrete_obj");
      super.new(name);
    endfunction
  endclass

  class pair_test extends uvm_test;
    `uvm_component_utils(pair_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      uvm_class_pair #(uvm_object, uvm_object) pair;
      concrete_obj a, b;
      phase.raise_objection(this);

      a = new("obj_a");
      b = new("obj_b");
      pair = new("pair");
      pair.first = a;
      pair.second = b;

      `uvm_info("TEST", $sformatf("pair first name: %s", pair.first.get_name()), UVM_LOW)
      `uvm_info("TEST", $sformatf("pair second name: %s", pair.second.get_name()), UVM_LOW)

      if (pair.first == a && pair.second == b)
        `uvm_info("TEST", "class pair: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "class pair: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("pair_test");
endmodule

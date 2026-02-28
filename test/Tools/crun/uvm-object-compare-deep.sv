// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test deep object comparison: nested objects, mismatch at deepest level.

// CHECK: [TEST] identical deep compare: PASS
// CHECK: [TEST] mismatch at leaf detected: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_leaf extends uvm_object;
    `uvm_object_utils(edge_leaf)
    int value;
    function new(string name = "edge_leaf");
      super.new(name);
    endfunction
    function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      edge_leaf rhs_;
      if (!$cast(rhs_, rhs)) return 0;
      return (value == rhs_.value);
    endfunction
  endclass

  class edge_mid extends uvm_object;
    `uvm_object_utils(edge_mid)
    edge_leaf leaf;
    function new(string name = "edge_mid");
      super.new(name);
      leaf = new("leaf");
    endfunction
    function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      edge_mid rhs_;
      if (!$cast(rhs_, rhs)) return 0;
      return leaf.compare(rhs_.leaf, comparer);
    endfunction
  endclass

  class edge_top_obj extends uvm_object;
    `uvm_object_utils(edge_top_obj)
    edge_mid mid;
    function new(string name = "edge_top_obj");
      super.new(name);
      mid = new("mid");
    endfunction
    function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      edge_top_obj rhs_;
      if (!$cast(rhs_, rhs)) return 0;
      return mid.compare(rhs_.mid, comparer);
    endfunction
  endclass

  class edge_deep_cmp_test extends uvm_test;
    `uvm_component_utils(edge_deep_cmp_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      edge_top_obj a, b;
      phase.raise_objection(this);

      a = new("a");
      b = new("b");
      a.mid.leaf.value = 42;
      b.mid.leaf.value = 42;

      // Identical
      if (a.compare(b))
        `uvm_info("TEST", "identical deep compare: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "identical deep compare: FAIL")

      // Mismatch at leaf
      b.mid.leaf.value = 99;
      if (!a.compare(b))
        `uvm_info("TEST", "mismatch at leaf detected: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "mismatch at leaf detected: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_deep_cmp_test");
endmodule

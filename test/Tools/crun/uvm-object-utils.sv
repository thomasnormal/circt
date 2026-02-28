// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_object utility methods: copy, clone, compare, print, convert2string.

// CHECK: [TEST] convert2string: PASS
// CHECK: [TEST] copy: PASS
// CHECK: [TEST] clone: PASS
// CHECK: [TEST] compare equal: PASS
// CHECK: [TEST] compare not-equal: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_txn extends uvm_sequence_item;
    `uvm_object_utils(my_txn)

    int addr;
    int data;
    string tag;

    function new(string name = "my_txn");
      super.new(name);
    endfunction

    virtual function string convert2string();
      return $sformatf("addr=%0d data=%0d tag=%s", addr, data, tag);
    endfunction

    virtual function void do_copy(uvm_object rhs);
      my_txn rhs_t;
      super.do_copy(rhs);
      if ($cast(rhs_t, rhs)) begin
        addr = rhs_t.addr;
        data = rhs_t.data;
        tag  = rhs_t.tag;
      end
    endfunction

    virtual function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      my_txn rhs_t;
      if (!$cast(rhs_t, rhs)) return 0;
      return (addr == rhs_t.addr) && (data == rhs_t.data) && (tag == rhs_t.tag);
    endfunction
  endclass

  class obj_utils_test extends uvm_test;
    `uvm_component_utils(obj_utils_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      my_txn a, b, c;

      phase.raise_objection(this);

      a = my_txn::type_id::create("a");
      a.addr = 100;
      a.data = 32'hDEAD;
      a.tag  = "hello";

      // Test 1: convert2string
      if (a.convert2string() == "addr=100 data=57005 tag=hello")
        `uvm_info("TEST", "convert2string: PASS", UVM_LOW)
      else
        `uvm_error("TEST", {"convert2string: FAIL — got: ", a.convert2string()})

      // Test 2: copy
      b = my_txn::type_id::create("b");
      b.copy(a);
      if (b.addr == 100 && b.data == 32'hDEAD && b.tag == "hello")
        `uvm_info("TEST", "copy: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "copy: FAIL")

      // Test 3: clone
      $cast(c, a.clone());
      if (c != null && c.addr == 100 && c.data == 32'hDEAD && c.tag == "hello")
        `uvm_info("TEST", "clone: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "clone: FAIL")

      // Test 4: compare (equal)
      if (a.compare(b))
        `uvm_info("TEST", "compare equal: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare equal: FAIL")

      // Test 5: compare (not equal)
      b.data = 999;
      if (!a.compare(b))
        `uvm_info("TEST", "compare not-equal: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare not-equal: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("obj_utils_test");
endmodule

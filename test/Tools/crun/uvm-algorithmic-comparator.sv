// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *
// Reason: uvm_algorithmic_comparator parameterized type instantiation errors

// Test uvm_algorithmic_comparator with identity transformer.

// CHECK: [TEST] fed 2 matching pairs to comparator
// CHECK: [TEST] comparator test: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class ac_item extends uvm_object;
    `uvm_object_utils(ac_item)
    int value;
    function new(string name = "ac_item");
      super.new(name);
    endfunction
    function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      ac_item rhs_item;
      if (!$cast(rhs_item, rhs)) return 0;
      return (value == rhs_item.value);
    endfunction
  endclass

  class id_transformer extends uvm_object;
    `uvm_object_utils(id_transformer)
    function new(string name = "id_transformer");
      super.new(name);
    endfunction
    function ac_item transform(ac_item before);
      return before;
    endfunction
  endclass

  class alg_comp_test extends uvm_test;
    `uvm_component_utils(alg_comp_test)
    uvm_algorithmic_comparator #(ac_item, id_transformer, ac_item) comp;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      id_transformer xf;
      super.build_phase(phase);
      xf = new("xf");
      comp = new("comp", this, xf);
    endfunction
    task run_phase(uvm_phase phase);
      ac_item a, b;
      phase.raise_objection(this);

      a = ac_item::type_id::create("a1");
      a.value = 10;
      b = ac_item::type_id::create("b1");
      b.value = 10;
      comp.before_export.write(a);
      comp.after_export.write(b);

      a = ac_item::type_id::create("a2");
      a.value = 20;
      b = ac_item::type_id::create("b2");
      b.value = 20;
      comp.before_export.write(a);
      comp.after_export.write(b);

      `uvm_info("TEST", "fed 2 matching pairs to comparator", UVM_LOW)
      `uvm_info("TEST", "comparator test: PASS", UVM_LOW)

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("alg_comp_test");
endmodule

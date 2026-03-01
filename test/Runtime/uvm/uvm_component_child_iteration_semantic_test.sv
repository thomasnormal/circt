// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
//
// CHECK: [TEST] child iteration count = 2
// CHECK: [TEST] child lookup by iterated key succeeded for all children
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps

`include "uvm_macros.svh"

package uvm_component_child_iteration_semantic_test_pkg;
  import uvm_pkg::*;

  class child_comp extends uvm_component;
    `uvm_component_utils(child_comp)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class parent_comp extends uvm_component;
    `uvm_component_utils(parent_comp)

    child_comp c0;
    child_comp c1;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      c0 = child_comp::type_id::create("c0", this);
      c1 = child_comp::type_id::create("c1", this);
    endfunction
  endclass
endpackage

module tb_top;
  import uvm_pkg::*;
  import uvm_component_child_iteration_semantic_test_pkg::*;

  initial begin
    parent_comp p;
    string child_name;
    int seen;
    bit all_lookup_ok;

    p = new("p", null);
    seen = 0;
    all_lookup_ok = 1;

    if (p.get_first_child(child_name)) begin
      do begin
        uvm_component c;
        seen++;
        c = p.get_child(child_name);
        if (c == null)
          all_lookup_ok = 0;
      end while (p.get_next_child(child_name));
    end

    $display("[TEST] child iteration count = %0d", seen);
    if (all_lookup_ok)
      $display("[TEST] child lookup by iterated key succeeded for all children");
    else
      $display("[TEST] child lookup by iterated key FAILED");

    $finish;
  end
endmodule

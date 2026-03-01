// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top tb_top --max-time=50000000000 2>&1 | FileCheck %s
//
// CHECK: [TEST] bracket0 null=0 eq=1
// CHECK: [TEST] bracket1 null=0 eq=1
// CHECK: [TEST] underscore0 null=0 eq=1
// CHECK: [TEST] underscore1 null=0 eq=1
// CHECK: [TEST] has_child bracket0=1 bracket1=1
// CHECK: [TEST] child count = 4
// CHECK: [TEST] all bracketed child lookups passed
// CHECK-NOT: UVM_FATAL

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class child_comp extends uvm_component;
  `uvm_component_utils(child_comp)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
endclass

class parent_comp extends uvm_component;
  `uvm_component_utils(parent_comp)

  child_comp bracket0;
  child_comp bracket1;
  child_comp underscore0;
  child_comp underscore1;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    bracket0 = child_comp::type_id::create("agent[0]", this);
    bracket1 = child_comp::type_id::create("agent[1]", this);
    underscore0 = child_comp::type_id::create("agent_0", this);
    underscore1 = child_comp::type_id::create("agent_1", this);
  endfunction
endclass

module tb_top;
  initial begin
    parent_comp p;
    uvm_component c;
    bit ok;
    bit has0;
    bit has1;

    p = new("p", null);
    ok = 1'b1;

    c = p.get_child("agent[0]");
    $display("[TEST] bracket0 null=%0d eq=%0d", c == null, c == p.bracket0);
    if (c == null || c != p.bracket0)
      ok = 1'b0;

    c = p.get_child("agent[1]");
    $display("[TEST] bracket1 null=%0d eq=%0d", c == null, c == p.bracket1);
    if (c == null || c != p.bracket1)
      ok = 1'b0;

    c = p.get_child("agent_0");
    $display("[TEST] underscore0 null=%0d eq=%0d",
             c == null, c == p.underscore0);
    if (c == null || c != p.underscore0)
      ok = 1'b0;

    c = p.get_child("agent_1");
    $display("[TEST] underscore1 null=%0d eq=%0d",
             c == null, c == p.underscore1);
    if (c == null || c != p.underscore1)
      ok = 1'b0;

    has0 = p.has_child("agent[0]");
    has1 = p.has_child("agent[1]");
    $display("[TEST] has_child bracket0=%0d bracket1=%0d", has0, has1);
    if (!(has0 && has1))
      ok = 1'b0;

    $display("[TEST] child count = %0d", p.get_num_children());
    if (p.get_num_children() != 4)
      ok = 1'b0;

    if (ok)
      $display("[TEST] all bracketed child lookups passed");
    else
      $fatal(1, "[TEST] bracketed child lookup failed");

    $finish;
  end
endmodule

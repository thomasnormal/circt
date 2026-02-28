//===----------------------------------------------------------------------===//
// UVM set_jump_phase(null) active-state regression
//===----------------------------------------------------------------------===//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=1000000000 2>&1 | FileCheck %s
//
// CHECK: BEFORE_SET_JUMP_NULL
// CHECK: AFTER_SET_JUMP_NULL
// CHECK-NOT: PH_BADJUMP

`timescale 1ns/1ps

import uvm_pkg::*;
`include "uvm_macros.svh"

class jump_null_test extends uvm_test;
  `uvm_component_utils(jump_null_test)

  function new(string name = "jump_null_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  virtual task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    $display("BEFORE_SET_JUMP_NULL");
    phase.set_jump_phase(null);
    $display("AFTER_SET_JUMP_NULL");
    phase.drop_objection(this);
  endtask
endclass

module top;
  initial run_test("jump_null_test");
endmodule

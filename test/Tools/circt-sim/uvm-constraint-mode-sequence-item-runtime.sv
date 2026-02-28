// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top --max-time 200000000 2>&1 | FileCheck %s

// Regression: disabling a named constraint on a uvm_sequence_item must affect
// subsequent randomize() calls. Inline randomize() with constraints must also
// constrain that call.
// CHECK: MODE_OK
// CHECK: INLINE_OK
// CHECK-NOT: MODE_FAIL
// CHECK-NOT: INLINE_FAIL

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class mem_item extends uvm_sequence_item;
  `uvm_object_utils(mem_item)

  rand bit we;
  rand logic [3:0] addr;
  constraint read_c { we == 0; }

  function new(string name = "mem_item");
    super.new(name);
  endfunction
endclass

module top;
  initial begin
    mem_item item = mem_item::type_id::create("item");
    int sawWrite = 0;

    item.read_c.constraint_mode(0);
    repeat (128) begin
      void'(item.randomize());
      if (item.we == 1'b1)
        sawWrite++;
    end
    if (sawWrite > 0)
      $display("MODE_OK");
    else
      $display("MODE_FAIL saw_write=%0d", sawWrite);

    void'(item.randomize() with { addr inside {4'd0, 4'd15}; we == 1'b1; });
    if (item.we == 1'b1 && (item.addr == 4'd0 || item.addr == 4'd15))
      $display("INLINE_OK");
    else
      $display("INLINE_FAIL we=%0d addr=%0d", item.we, item.addr);

    $finish;
  end
endmodule

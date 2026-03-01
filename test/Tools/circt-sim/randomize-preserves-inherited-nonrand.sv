// Test that randomize() preserves inherited non-rand base-class fields.
// Bug: MooreToCore preserved only properties declared in the most-derived
// class, so inherited state could be clobbered by randomize_basic.
//
// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

class base_item;
  int seq_id;
endclass

class item extends base_item;
  rand bit [7:0] data;
  constraint c { data inside {[8'h10:8'hF0]}; }
endclass

module top;
  initial begin
    item it;
    int ok;
    it = new;
    it.seq_id = 1234;
    ok = it.randomize();
    if (!ok) begin
      $display("FAIL randomize");
      $finish;
    end
    if (it.seq_id != 1234) begin
      $display("FAIL seq_id=%0d", it.seq_id);
      $finish;
    end
    $display("PASS data=%0h seq_id=%0d", it.data, it.seq_id);
    $finish;
  end
endmodule

// CHECK: PASS data=
// CHECK-NOT: FAIL
// CHECK: Simulation completed

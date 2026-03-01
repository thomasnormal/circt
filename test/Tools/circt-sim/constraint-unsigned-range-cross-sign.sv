// Test that unsigned range constraints crossing the sign bit are satisfiable.
// Bug: constants were sign-extended during range extraction, turning
//      [8'h10:8'hF0] into [16:-16], which was treated as infeasible.
//
// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

module top;
  class item;
    rand bit [7:0] data;
    constraint c { data inside {[8'h10:8'hF0]}; }
  endclass

  initial begin
    item it;
    int ok;
    it = new;
    ok = it.randomize();
    if (!ok) begin
      $display("FAIL randomize");
      $finish;
    end
    if (!(it.data >= 8'h10 && it.data <= 8'hF0)) begin
      $display("FAIL out-of-range data=%0h", it.data);
      $finish;
    end
    $display("PASS data=%0h", it.data);
    $finish;
  end
endmodule

// CHECK: PASS data=
// CHECK-NOT: FAIL
// CHECK: Simulation completed

// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that unsatisfiable cross-variable constraints cause randomize() to
// still complete (possibly returning failure).
class unsat;
  rand bit [3:0] x;
  rand bit [3:0] y;
  constraint c1 { x >= 14; }
  constraint c2 { y >= 14; }
  constraint c3 { x + y < 10; }

  function new();
    x = 0;
    y = 0;
  endfunction
endclass

module top;
  initial begin
    unsat obj = new();
    int ok;
    ok = obj.randomize();
    // Randomize should still complete without crashing
    // CHECK: randomize_called=1
    $display("randomize_called=1");
    $finish;
  end
endmodule

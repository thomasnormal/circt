// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test case equality (===, !==) and wildcard equality (==?, !=?)
module top;
  logic [3:0] a, b;

  initial begin
    a = 4'b1010;
    b = 4'b1010;

    // Case equality
    // CHECK: case_eq=1
    $display("case_eq=%0d", a === b);

    b = 4'b1011;
    // CHECK: case_neq=1
    $display("case_neq=%0d", a !== b);

    // Wildcard equality (? matches X or Z)
    a = 4'b10x0;
    b = 4'b1010;
    // ==? treats X/Z in RHS as wildcards
    // CHECK: wild_eq=1
    $display("wild_eq=%0d", b ==? a);

    $finish;
  end
endmodule

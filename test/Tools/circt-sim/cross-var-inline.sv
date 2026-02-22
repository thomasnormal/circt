// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that inline cross-variable constraint with randomize() with { ... }
// is properly enforced.
class basic;
  rand bit [7:0] a;
  rand bit [7:0] b;

  function new();
    a = 0;
    b = 0;
  endfunction
endclass

module top;
  initial begin
    basic obj = new();
    int ok;
    int valid = 1;
    int i;
    for (i = 0; i < 100; i = i + 1) begin
      ok = obj.randomize() with { a + b < 50; };
      if (obj.a + obj.b >= 50)
        valid = 0;
    end
    // CHECK: inline_cross=1
    $display("inline_cross=%0d", valid);
    $finish;
  end
endmodule

// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// CHECK: SVA_PASS: local var compound init
// CHECK-NOT: SVA assertion failed

module top;
  bit clk = 0;
  always #5 clk = ~clk;

  property p;
    int v = 0;
    (1, v += 1) ##1 (v == 1);
  endproperty

  a_local_var_compound_init: assert property (@(posedge clk) p);

  initial begin
    repeat (6) @(posedge clk);
    $display("SVA_PASS: local var compound init");
    $finish;
  end
endmodule

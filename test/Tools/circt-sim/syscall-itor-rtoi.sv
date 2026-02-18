// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $itor (integer to real) and $rtoi (real to integer)
module top;
  integer i;
  real r;

  initial begin
    i = 42;
    r = $itor(i);
    // CHECK: itor=42.000000
    $display("itor=%f", r);

    r = 3.7;
    i = $rtoi(r);
    // CHECK: rtoi=3
    $display("rtoi=%0d", i);

    r = -2.9;
    i = $rtoi(r);
    // CHECK: rtoi_neg=-2
    $display("rtoi_neg=%0d", i);

    $finish;
  end
endmodule

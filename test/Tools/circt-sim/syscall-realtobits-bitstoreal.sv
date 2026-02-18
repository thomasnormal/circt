// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $realtobits and $bitstoreal
module top;
  real r;
  reg [63:0] bits;
  real r2;

  initial begin
    r = 1.0;
    bits = $realtobits(r);
    r2 = $bitstoreal(bits);
    // CHECK: roundtrip=1.000000
    $display("roundtrip=%f", r2);

    r = -42.5;
    bits = $realtobits(r);
    r2 = $bitstoreal(bits);
    // CHECK: roundtrip_neg=-42.500000
    $display("roundtrip_neg=%f", r2);

    $finish;
  end
endmodule

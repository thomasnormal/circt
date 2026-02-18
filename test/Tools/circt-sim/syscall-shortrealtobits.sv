// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $shortrealtobits and $bitstoshortreal
module top;
  shortreal sr;
  reg [31:0] bits;
  shortreal sr2;

  initial begin
    sr = 1.0;
    bits = $shortrealtobits(sr);
    sr2 = $bitstoshortreal(bits);
    // CHECK: sr_roundtrip=1.000000
    $display("sr_roundtrip=%f", sr2);

    sr = -3.14;
    bits = $shortrealtobits(sr);
    sr2 = $bitstoshortreal(bits);
    // CHECK: sr_neg=-3.14
    $display("sr_neg=%.2f", sr2);

    $finish;
  end
endmodule

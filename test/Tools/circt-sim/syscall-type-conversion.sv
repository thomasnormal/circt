// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test type conversion: int', shortint', longint', byte', bit, logic conversions
module top;
  int i;
  shortint si;
  longint li;
  byte b;
  real r;

  initial begin
    // Integer to byte (truncation)
    i = 300;
    b = byte'(i);
    // CHECK: byte_trunc=44
    $display("byte_trunc=%0d", b);  // 300 mod 256 = 44

    // Shortint range
    si = shortint'(32768);
    // CHECK: shortint=-32768
    $display("shortint=%0d", si);  // wraps to -32768

    // Real to int
    r = 42.9;
    i = int'(r);
    // CHECK: real_to_int=42
    $display("real_to_int=%0d", i);

    $finish;
  end
endmodule

// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $display format specifiers: %d, %h, %o, %b, %s, %0d
module top;
  integer i;
  string s;

  initial begin
    i = 255;

    // Decimal
    // CHECK: dec=255
    $display("dec=%0d", i);

    // Hex
    // CHECK: hex=ff
    $display("hex=%0h", i);

    // Octal
    // CHECK: oct=377
    $display("oct=%0o", i);

    // Binary
    // CHECK: bin=11111111
    $display("bin=%0b", i);

    // String
    s = "hello";
    // CHECK: str=hello
    $display("str=%s", s);

    // Multiple format args in one call — verifies argument interleaving
    i = 42;
    s = "world";
    // CHECK: multi=42 world ff
    $display("multi=%0d %s %0h", i, s, 255);

    // Negative decimal
    i = -1;
    // CHECK: neg=-1
    $display("neg=%0d", i);

    // Zero padding
    i = 42;
    // CHECK: padded=002a
    $display("padded=%04h", i);

    $finish;
  end
endmodule

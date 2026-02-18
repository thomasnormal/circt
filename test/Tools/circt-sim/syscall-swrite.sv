// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $swrite and $sformat — format into string variable
module top;
  string s;
  integer i;

  initial begin
    i = 42;

    // $swrite formats into a string
    $swrite(s, "value=%0d hex=%0h", i, i);
    // CHECK: swrite=value=42 hex=2a
    $display("swrite=%s", s);

    // $sformat with explicit format
    $sformat(s, "num=%0d", 255);
    // CHECK: sformat=num=255
    $display("sformat=%s", s);

    // Empty format
    $swrite(s, "literal_only");
    // CHECK: literal=literal_only
    $display("literal=%s", s);

    $finish;
  end
endmodule

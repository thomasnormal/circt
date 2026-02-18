// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $display, $write, $strobe, $monitor (basic output)
module top;
  integer x;

  initial begin
    x = 42;

    // $display adds newline
    // CHECK: display=42
    $display("display=%0d", x);

    // $write does NOT add newline
    $write("write1=");
    $write("%0d", x);
    // CHECK: write1=42
    $write("\n");

    // $displayb, $displayo, $displayh
    // CHECK: displayh=2a
    $displayh("displayh=", x);

    // CHECK: displayo=52
    $displayo("displayo=", x);

    // CHECK: displayb=00000000000000000000000000101010
    $displayb("displayb=", x);

    $finish;
  end
endmodule

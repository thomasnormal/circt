// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  integer count, ival;
  reg [8*20-1:0] sval;

  initial begin
    // Basic integer + string scan
    count = $sscanf("42 hello", "%d %s", ival, sval);
    // CHECK: count=2 ival=42 sval=hello
    $display("count=%0d ival=%0d sval=%s", count, ival, sval);

    // Hex scan
    count = $sscanf("ff", "%h", ival);
    // CHECK: count=1 ival=255
    $display("count=%0d ival=%0d", count, ival);

    // Multiple integers
    count = $sscanf("10 20 30", "%d %d %d", ival, count, ival);
    // CHECK: sscanf3=3
    $display("sscanf3=%0d", count);

    $finish;
  end
endmodule

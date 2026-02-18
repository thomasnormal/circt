// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $clog2 system function
module top;
  initial begin
    // CHECK: clog2_1=0
    $display("clog2_1=%0d", $clog2(1));
    // CHECK: clog2_2=1
    $display("clog2_2=%0d", $clog2(2));
    // CHECK: clog2_3=2
    $display("clog2_3=%0d", $clog2(3));
    // CHECK: clog2_4=2
    $display("clog2_4=%0d", $clog2(4));
    // CHECK: clog2_5=3
    $display("clog2_5=%0d", $clog2(5));
    // CHECK: clog2_8=3
    $display("clog2_8=%0d", $clog2(8));
    // CHECK: clog2_256=8
    $display("clog2_256=%0d", $clog2(256));
    // CHECK: clog2_0=0
    $display("clog2_0=%0d", $clog2(0));
    $finish;
  end
endmodule

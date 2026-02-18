// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  string s;

  initial begin
    // $sformat writes formatted string to first argument
    $sformat(s, "x=%0d y=%0d", 10, 20);
    // CHECK: result=x=10 y=20
    $display("result=%s", s);

    $sformat(s, "hex=%h", 16'hCAFE);
    // CHECK: result2=hex=cafe
    $display("result2=%s", s);

    $finish;
  end
endmodule

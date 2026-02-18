// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  string s;

  initial begin
    // $psprintf is an alias for $sformatf
    s = $psprintf("val=%0d hex=%0h", 100, 255);
    // CHECK: result=val=100 hex=ff
    $display("result=%s", s);
    $finish;
  end
endmodule

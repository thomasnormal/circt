// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  string s;
  integer val;

  initial begin
    // Test decimal format
    s = $sformatf("val=%0d", 42);
    // CHECK: result1=val=42
    $display("result1=%s", s);

    // Test hex format
    s = $sformatf("hex=%0h", 255);
    // CHECK: result2=hex=ff
    $display("result2=%s", s);

    // Test binary format
    s = $sformatf("bin=%b", 8'hA5);
    // CHECK: result3=bin=10100101
    $display("result3=%s", s);

    // Test string format
    s = $sformatf("name=%s", "hello");
    // CHECK: result4=name=hello
    $display("result4=%s", s);

    // Test multiple args
    s = $sformatf("%0d+%0d=%0d", 3, 4, 7);
    // CHECK: result5=3+4=7
    $display("result5=%s", s);

    $finish;
  end
endmodule

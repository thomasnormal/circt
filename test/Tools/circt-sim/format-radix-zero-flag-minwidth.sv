// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --mode interpret --top top 2>&1 | FileCheck %s

module top;
  logic [5:0] a;
  logic [7:0] b;

  initial begin
    a = 6'b000001;
    b = 8'bz0zz0z01;

    $display("o<%o>", a);
    $display("0o<%0o>", a);
    $display("h<%h>", a);
    $display("0h<%0h>", a);

    $display("oz<%o>", b);
    $display("0oz<%0o>", b);
    $display("05oz<%05o>", b);

    // CHECK: o<01>
    // CHECK: 0o<1>
    // CHECK: h<01>
    // CHECK: 0h<1>
    // CHECK: oz<ZZZ>
    // CHECK: 0oz<Z>
    // CHECK: 05oz<00ZZZ>
    $finish;
  end
endmodule

// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test concatenation and replication operators
module top;
  logic [3:0] a, b;
  logic [7:0] c;
  logic [15:0] d;

  initial begin
    // Concatenation
    a = 4'hA;
    b = 4'h5;
    c = {a, b};
    // CHECK: concat=a5
    $display("concat=%h", c);

    // Replication
    a = 4'hF;
    d = {4{a}};
    // CHECK: repl=ffff
    $display("repl=%h", d);

    // Concatenation with different widths
    c = {4'b1100, 4'b0011};
    // CHECK: mixed=c3
    $display("mixed=%h", c);

    // Nested concatenation
    d = {{2{4'hA}}, {2{4'h5}}};
    // CHECK: nested=aa55
    $display("nested=%h", d);

    $finish;
  end
endmodule

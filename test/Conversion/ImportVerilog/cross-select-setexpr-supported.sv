// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectSetExprSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    coverpoint a { bins xa[] = {[0:1]}; }
    coverpoint b { bins xb[] = {[0:1]}; }
    X: cross a, b {
      bins one = '{ '{1, 0}, '{0, 1} };
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @one kind<bins> {
// CHECK:   moore.binsof @a intersect [1]
// CHECK:   moore.binsof @b intersect [0]
// CHECK:   moore.binsof @a intersect [0] {group = 1 : i32}
// CHECK:   moore.binsof @b intersect [1] {group = 1 : i32}
// CHECK: }

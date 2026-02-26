// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectSetExprFunctionLiteralSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    coverpoint a;
    coverpoint b;
    X: cross a, b {
      function CrossQueueType mk();
        mk = '{ '{1, 0}, '{0, 1} };
      endfunction
      bins one = mk();
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @one kind<bins> {
// CHECK:   moore.binsof @a intersect [1]
// CHECK:   moore.binsof @b intersect [0]
// CHECK:   moore.binsof @a intersect [0] {group = 1 : i32}
// CHECK:   moore.binsof @b intersect [1] {group = 1 : i32}
// CHECK: }

// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectSetExprFunctionAssignCallThenPushbackSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    coverpoint a;
    coverpoint b;
    X: cross a, b {
      function CrossQueueType h1();
        h1.push_back('{0, 0});
      endfunction
      function CrossQueueType mk(int lim);
        mk = h1();
        mk.push_back('{1, 1});
      endfunction
      bins one = mk(1);
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @one kind<bins> {
// CHECK:   moore.binsof @a intersect [0]
// CHECK:   moore.binsof @b intersect [0]
// CHECK:   moore.binsof @a intersect [1] {group = 1 : i32}
// CHECK:   moore.binsof @b intersect [1] {group = 1 : i32}
// CHECK: }

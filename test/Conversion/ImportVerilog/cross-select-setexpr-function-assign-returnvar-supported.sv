// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectSetExprFunctionAssignReturnvarSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    coverpoint a;
    coverpoint b;
    X: cross a, b {
      function CrossQueueType mk();
        CrossQueueType q;
        q.push_back('{0, 0});
        q.push_back('{1, 1});
        mk = q;
      endfunction
      bins one = mk();
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @one kind<bins> {
// CHECK:   moore.binsof @a intersect [0]
// CHECK:   moore.binsof @b intersect [0]
// CHECK:   moore.binsof @a intersect [1] {group = 1 : i32}
// CHECK:   moore.binsof @b intersect [1] {group = 1 : i32}
// CHECK: }

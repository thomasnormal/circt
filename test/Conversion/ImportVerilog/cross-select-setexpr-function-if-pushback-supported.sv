// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectSetExprFunctionIfPushbackSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    coverpoint a;
    coverpoint b;
    X: cross a, b {
      function CrossQueueType mk(int lim);
        int i;
        i = 0;
        if (lim > 0)
          mk.push_back('{i, i});
        if (lim > 1)
          mk.push_back('{i + 1, i + 1});
      endfunction
      bins one = mk(2);
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @one kind<bins> {
// CHECK:   moore.binsof @a intersect [0]
// CHECK:   moore.binsof @b intersect [0]
// CHECK:   moore.binsof @a intersect [1] {group = 1 : i32}
// CHECK:   moore.binsof @b intersect [1] {group = 1 : i32}
// CHECK: }

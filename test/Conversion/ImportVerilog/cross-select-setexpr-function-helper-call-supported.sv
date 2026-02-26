// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectSetExprFunctionHelperCallSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    coverpoint a;
    coverpoint b;
    X: cross a, b {
      function CrossQueueType sub(int lim);
        int i;
        for (i = 0; i < lim; ++i)
          sub.push_back('{i, i});
      endfunction

      function CrossQueueType mk(int lim);
        return sub(lim);
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

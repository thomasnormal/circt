// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectSetExprFunctionAssignConditionalCallSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    coverpoint a;
    coverpoint b;
    X: cross a, b {
      function CrossQueueType h1();
        h1.push_back('{0, 0});
      endfunction
      function CrossQueueType h2();
        h2.push_back('{1, 1});
      endfunction
      function CrossQueueType mk(int lim);
        mk = (lim > 0) ? h1() : h2();
      endfunction
      bins one = mk(1);
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @one kind<bins> {
// CHECK:   moore.binsof @a intersect [0]
// CHECK:   moore.binsof @b intersect [0]
// CHECK: }

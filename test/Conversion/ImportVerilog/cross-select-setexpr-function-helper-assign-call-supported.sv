// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectSetExprFunctionHelperAssignCallSupported;
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
        mk = sub(lim);
      endfunction

      bins one = mk(2);
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @one kind<bins> {
// CHECK:   moore.binsof @a negate
// CHECK: }

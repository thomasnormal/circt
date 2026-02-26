// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectSetExprFunctionDisableBlockSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    coverpoint a;
    coverpoint b;
    X: cross a, b {
      function CrossQueueType mk(int lim);
        begin : blk
          mk.push_back('{0, 0});
          if (lim > 0)
            disable blk;
          mk.push_back('{1, 1});
        end
      endfunction
      bins one = mk(1);
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @one kind<bins> {
// CHECK:   moore.binsof @a intersect [0]
// CHECK:   moore.binsof @b intersect [0]
// CHECK: }

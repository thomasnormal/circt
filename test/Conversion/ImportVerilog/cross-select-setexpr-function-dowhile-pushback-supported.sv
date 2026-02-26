// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectSetExprFunctionDoWhilePushbackSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    coverpoint a;
    coverpoint b;
    X: cross a, b {
      function CrossQueueType mk(int lim);
        int i;
        i = 0;
        do begin
          mk.push_back('{i, i});
          i = i + 1;
        end while (i < lim);
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

// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectSetExprFunctionForLoopInitAssignCallSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    coverpoint a;
    coverpoint b;
    X: cross a, b {
      function CrossQueueType h1();
        h1.push_back('{0, 0});
      endfunction
      function CrossQueueType mk();
        int i;
        for (i = 0, mk = h1(); i < 1; i++) begin
        end
      endfunction
      bins one = mk();
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @one kind<bins> {
// CHECK:   moore.binsof @a intersect [0]
// CHECK:   moore.binsof @b intersect [0]
// CHECK: }

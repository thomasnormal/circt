// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectSetExprFunctionForLoopStepAssignCallSupported;
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
      function CrossQueueType mk();
        int i;
        mk = h1();
        for (i = 0; i < 1; i++, mk = h2()) begin
        end
      endfunction
      bins one = mk();
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @one kind<bins> {
// CHECK:   moore.binsof @a intersect [1]
// CHECK:   moore.binsof @b intersect [1]
// CHECK: }

// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectCrossIdSupported;
  bit clk;
  int a, b;

  // CHECK: moore.covercross.decl @X targets [@a, @b] {
  // CHECK:   moore.crossbin.decl @all_pairs kind<bins> {
  // CHECK-NEXT:   }
  // CHECK: }
  covergroup cg @(posedge clk);
    coverpoint a;
    coverpoint b;
    X: cross a, b {
      bins all_pairs = X;
    }
  endgroup
endmodule

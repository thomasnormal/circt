// RUN: circt-verilog --language-version 1800-2023 %s --ir-moore | FileCheck %s

// Regression: keep cross-select intersect lowering when the range/value comes
// from elaboration-stable initialized symbols.
module CrossSelectIntersectInitializedVarSupported;
  bit clk;
  bit [3:0] a, b;

  int lo = 1;
  int hi = 2;
  int singleton = 3;

  covergroup cg @(posedge clk);
    a_cp: coverpoint a;
    b_cp: coverpoint b;
    X: cross a_cp, b_cp {
      bins in_range = binsof(a_cp) intersect {[lo:hi]} && binsof(b_cp);
      bins exact = binsof(a_cp) intersect {singleton} && binsof(b_cp);
    }
  endgroup

  cg c = new();

  // Exercise sampling paths so the construct is used functionally, not just
  // parsed.
  initial begin
    a = 4'd2;
    b = 4'd1;
    clk = 1'b0;
    #1 clk = 1'b1;
    c.sample();
    #1 clk = 1'b0;
  end
endmodule

// CHECK: moore.crossbin.decl @in_range kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [1, 2]
// CHECK:   moore.binsof @b_cp
// CHECK: }
// CHECK: moore.crossbin.decl @exact kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [3]
// CHECK:   moore.binsof @b_cp
// CHECK: }

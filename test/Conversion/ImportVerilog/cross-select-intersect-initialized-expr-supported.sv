// RUN: circt-verilog --language-version 1800-2023 %s --ir-moore | FileCheck %s

// Regression: cross-select intersect ranges must accept elaboration-stable
// initialized expression forms, not just bare initialized symbols.
module CrossSelectIntersectInitializedExprSupported;
  bit clk;
  bit [3:0] a, b;

  int GLO = 1;
  int GHI = 5;
  int ARR [0:1] = '{3, 6};

  covergroup cg @(posedge clk);
    a_cp: coverpoint a;
    b_cp: coverpoint b;
    X: cross a_cp, b_cp {
      bins arithmetic = binsof(a_cp) intersect {[GLO + 1:GHI - 1]} && binsof(b_cp);
      bins array_idx = binsof(a_cp) intersect {[ARR[0]:ARR[1]]} && binsof(b_cp);
      bins mixed = binsof(a_cp) intersect {[ARR[0] + 1:ARR[1] - 1]} && binsof(b_cp);
    }
  endgroup

  cg c = new();

  initial begin
    a = 4'd4;
    b = 4'd1;
    clk = 1'b0;
    #1 clk = 1'b1;
    c.sample();
    #1 clk = 1'b0;
  end
endmodule

// CHECK: moore.crossbin.decl @arithmetic kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [2, 3, 4]
// CHECK:   moore.binsof @b_cp
// CHECK: }
// CHECK: moore.crossbin.decl @array_idx kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [3, 4, 5, 6]
// CHECK:   moore.binsof @b_cp
// CHECK: }
// CHECK: moore.crossbin.decl @mixed kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [4, 5]
// CHECK:   moore.binsof @b_cp
// CHECK: }

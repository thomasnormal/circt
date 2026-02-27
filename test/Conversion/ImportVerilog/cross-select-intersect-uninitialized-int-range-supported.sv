// RUN: circt-verilog --language-version 1800-2023 %s --ir-moore | FileCheck %s

// Regression: simple intersect ranges over uninitialized 2-state ints are
// elaboration-stable in xrun (default 0) and should lower in ImportVerilog.
module CrossSelectIntersectUninitializedIntRangeSupported;
  bit clk;
  bit [3:0] a, b;

  int lo;
  int hi;
  integer ilo;
  integer ihi;
  logic [3:0] llo;
  logic [3:0] lhi;
  int iarr [0:1];
  integer iiarr [0:1];
  logic [3:0] llarr [0:1];

  covergroup cg @(posedge clk);
    a_cp: coverpoint a;
    b_cp: coverpoint b;
    X: cross a_cp, b_cp {
      bins uninit_range = binsof(a_cp) intersect {[lo:hi]} && binsof(b_cp);
      bins uninit_expr = binsof(a_cp) intersect {[lo + 1:hi + 2]} && binsof(b_cp);
      bins uninit_integer = binsof(a_cp) intersect {[ilo:ihi]} && binsof(b_cp);
      bins uninit_logic = binsof(a_cp) intersect {[llo:lhi]} && binsof(b_cp);
      bins uninit_int_array = binsof(a_cp) intersect {[iarr[0]:iarr[1]]} && binsof(b_cp);
      bins uninit_integer_array = binsof(a_cp) intersect {[iiarr[0]:iiarr[1]]} && binsof(b_cp);
      bins uninit_logic_array = binsof(a_cp) intersect {[llarr[0]:llarr[1]]} && binsof(b_cp);
    }
  endgroup

  cg c = new();

  initial begin
    a = 4'd1;
    b = 4'd3;
    clk = 1'b0;
    #1 clk = 1'b1;
    c.sample();
    #1 clk = 1'b0;
  end
endmodule

// CHECK: moore.crossbin.decl @uninit_range kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [0]
// CHECK:   moore.binsof @b_cp
// CHECK: }
// CHECK: moore.crossbin.decl @uninit_expr kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [1, 2]
// CHECK:   moore.binsof @b_cp
// CHECK: }
// CHECK: moore.crossbin.decl @uninit_integer kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [0]
// CHECK:   moore.binsof @b_cp
// CHECK: }
// CHECK: moore.crossbin.decl @uninit_logic kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [0]
// CHECK:   moore.binsof @b_cp
// CHECK: }
// CHECK: moore.crossbin.decl @uninit_int_array kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [0]
// CHECK:   moore.binsof @b_cp
// CHECK: }
// CHECK: moore.crossbin.decl @uninit_integer_array kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [0]
// CHECK:   moore.binsof @b_cp
// CHECK: }
// CHECK: moore.crossbin.decl @uninit_logic_array kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [0]
// CHECK:   moore.binsof @b_cp
// CHECK: }

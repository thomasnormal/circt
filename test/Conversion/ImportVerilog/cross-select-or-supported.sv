// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectOrSupported;
  bit clk;
  bit [3:0] a, b;

  covergroup cg @(posedge clk);
    cp_a: coverpoint a;
    cp_b: coverpoint b;
    x: cross cp_a, cp_b {
      bins row_or_col_0 = binsof(cp_a) intersect {0} || binsof(cp_b) intersect {0};
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @row_or_col_0 kind<bins> {
// CHECK:   moore.binsof @cp_a intersect [0]
// CHECK:   moore.binsof @cp_b intersect [0] {group = 1 : i32}
// CHECK: }

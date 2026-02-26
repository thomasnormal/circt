// RUN: circt-verilog --language-version 1800-2023 %s --ir-moore | FileCheck %s

module CrossSelectIntersectOpenRangeSupported;
  bit clk;
  bit [3:0] a, b;

  covergroup cg @(posedge clk);
    a_cp: coverpoint a;
    b_cp: coverpoint b;
    X: cross a_cp, b_cp {
      bins c = binsof(a_cp) intersect {[4:$]};
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @c kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// CHECK: }

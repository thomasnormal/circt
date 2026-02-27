// RUN: circt-verilog --language-version 1800-2023 %s --ir-moore | FileCheck %s

module CrossSelectIntersectOpenRangeUnsupported;
  bit clk;
  int a, b;

  covergroup cg @(posedge clk);
    coverpoint a;
    coverpoint b;
    X: cross a, b {
      bins c = binsof(a) intersect {[0:$]};
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @c kind<bins> {
// CHECK:   moore.binsof @a intersect_ranges [0, 9223372036854775807]
// CHECK: }

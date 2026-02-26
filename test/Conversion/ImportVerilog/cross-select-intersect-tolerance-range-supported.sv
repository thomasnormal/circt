// RUN: circt-verilog --language-version 1800-2023 %s --ir-moore | FileCheck %s

module CrossSelectIntersectToleranceRangeSupported;
  bit clk;
  int a, b;

  covergroup cg @(posedge clk);
    a_cp: coverpoint a;
    b_cp: coverpoint b;
    X: cross a_cp, b_cp {
      bins abs_tol = binsof(a_cp) intersect {[8 +/- 3]};
      bins rel_tol = binsof(a_cp) intersect {[8 +%- 25]};
    }
  endgroup
endmodule

// CHECK-LABEL: moore.crossbin.decl @abs_tol kind<bins> {
// CHECK:         moore.binsof @a_cp intersect [5, 6, 7, 8, 9, 10, 11]
// CHECK:       }
// CHECK-LABEL: moore.crossbin.decl @rel_tol kind<bins> {
// CHECK:         moore.binsof @a_cp intersect [6, 7, 8, 9, 10]
// CHECK:       }

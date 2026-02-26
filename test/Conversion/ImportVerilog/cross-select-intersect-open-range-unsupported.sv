// RUN: ! circt-verilog --language-version 1800-2023 %s --ir-moore 2>&1 | FileCheck %s

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

// CHECK: error: unsupported non-constant intersect value range in cross select expression

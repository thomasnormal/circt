// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectWithCoverpointSetTargetSupported;
  bit clk;
  bit [1:0] a, b;
  int vals[] = '{0, 2};

  covergroup cg @(posedge clk);
    a_cp: coverpoint a {
      bins even = vals;
    }
    b_cp: coverpoint b;
    X: cross a_cp, b_cp {
      bins sel = (binsof(a_cp.even)) with (a_cp > 1);
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @sel kind<bins> {
// CHECK:   moore.binsof @a_cp::@even
// CHECK:   moore.binsof @b_cp intersect [0]
// CHECK:   moore.binsof @a_cp::@even {group = 1 : i32}
// CHECK:   moore.binsof @b_cp intersect [1] {group = 1 : i32}
// CHECK:   moore.binsof @a_cp::@even {group = 2 : i32}
// CHECK:   moore.binsof @b_cp intersect [2] {group = 2 : i32}
// CHECK:   moore.binsof @a_cp::@even {group = 3 : i32}
// CHECK:   moore.binsof @b_cp intersect [3] {group = 3 : i32}
// CHECK: }

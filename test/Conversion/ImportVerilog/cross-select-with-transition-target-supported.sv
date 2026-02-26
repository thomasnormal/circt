// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectWithTransitionTargetSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    a_cp: coverpoint a {
      bins n = {[0:1]};
      bins tr = (2 => 3);
    }
    b_cp: coverpoint b;
    X: cross a_cp, b_cp {
      bins sel = (binsof(a_cp.tr)) with (a_cp > 1);
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @sel kind<bins> {
// CHECK:   moore.binsof @a_cp::@tr
// CHECK:   moore.binsof @b_cp intersect [0]
// CHECK:   moore.binsof @a_cp::@tr {group = 1 : i32}
// CHECK:   moore.binsof @b_cp intersect [1] {group = 1 : i32}
// CHECK:   moore.binsof @a_cp::@tr {group = 2 : i32}
// CHECK:   moore.binsof @b_cp intersect [2] {group = 2 : i32}
// CHECK:   moore.binsof @a_cp::@tr {group = 3 : i32}
// CHECK:   moore.binsof @b_cp intersect [3] {group = 3 : i32}
// CHECK: }

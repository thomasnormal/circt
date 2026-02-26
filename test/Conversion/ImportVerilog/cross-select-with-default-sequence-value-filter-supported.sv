// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectWithDefaultSequenceValueFilterSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    a_cp: coverpoint a {
      bins tr = (0 => 1);
      bins ds = default sequence;
    }
    b_cp: coverpoint b;
    X: cross a_cp, b_cp {
      bins sel = (binsof(a_cp.ds)) with (a_cp > 0);
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @sel kind<bins> {
// CHECK:   moore.binsof @a_cp::@ds
// CHECK:   moore.binsof @b_cp intersect [0]
// CHECK:   moore.binsof @a_cp::@ds {group = 1 : i32}
// CHECK:   moore.binsof @b_cp intersect [1] {group = 1 : i32}
// CHECK:   moore.binsof @a_cp::@ds {group = 2 : i32}
// CHECK:   moore.binsof @b_cp intersect [2] {group = 2 : i32}
// CHECK:   moore.binsof @a_cp::@ds {group = 3 : i32}
// CHECK:   moore.binsof @b_cp intersect [3] {group = 3 : i32}
// CHECK: }

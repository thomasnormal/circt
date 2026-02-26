// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectWithNestedOrSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    coverpoint a;
    coverpoint b;
    X: cross a, b {
      bins sel = binsof(a) with (a == 0) || binsof(b) with (b == 1);
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @sel kind<bins> {
// CHECK:   moore.binsof @a intersect [0]
// CHECK:   moore.binsof @b intersect [0]
// CHECK:   moore.binsof @a intersect [0] {group = 1 : i32}
// CHECK:   moore.binsof @b intersect [1] {group = 1 : i32}
// CHECK:   moore.binsof @a intersect [0] {group = 2 : i32}
// CHECK:   moore.binsof @b intersect [2] {group = 2 : i32}
// CHECK:   moore.binsof @a intersect [0] {group = 3 : i32}
// CHECK:   moore.binsof @b intersect [3] {group = 3 : i32}
// CHECK:   moore.binsof @a intersect [1] {group = 4 : i32}
// CHECK:   moore.binsof @b intersect [1] {group = 4 : i32}
// CHECK:   moore.binsof @a intersect [2] {group = 5 : i32}
// CHECK:   moore.binsof @b intersect [1] {group = 5 : i32}
// CHECK:   moore.binsof @a intersect [3] {group = 6 : i32}
// CHECK:   moore.binsof @b intersect [1] {group = 6 : i32}
// CHECK: }

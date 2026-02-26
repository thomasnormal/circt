// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectWithExplicitBinsSupported;
  bit clk;
  bit [1:0] a, b;

  covergroup cg @(posedge clk);
    cp_a: coverpoint a {
      bins lo = {[0:1]};
      bins hi = {[2:3]};
    }
    cp_b: coverpoint b {
      bins lo = {[0:1]};
      bins hi = {[2:3]};
    }
    X: cross cp_a, cp_b {
      bins sel = X with (cp_a + cp_b < 3);
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @sel kind<bins> {
// CHECK:   moore.binsof @cp_a::@lo
// CHECK:   moore.binsof @cp_b::@lo
// CHECK:   moore.binsof @cp_a::@lo {group = 1 : i32}
// CHECK:   moore.binsof @cp_b::@hi {group = 1 : i32}
// CHECK:   moore.binsof @cp_a::@hi {group = 2 : i32}
// CHECK:   moore.binsof @cp_b::@lo {group = 2 : i32}
// CHECK-NOT: {group = 3 : i32}
// CHECK: }

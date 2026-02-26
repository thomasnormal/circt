// RUN: circt-verilog %s --ir-moore | FileCheck %s

module CrossSelectWithWideAutoDomainSupported;
  bit clk;
  bit [8:0] a;
  bit b;

  covergroup cg @(posedge clk);
    a_cp: coverpoint a;
    b_cp: coverpoint b;
    X: cross a_cp, b_cp {
      bins sel = X with (a_cp == 9'd511 && b_cp == 1'b1);
    }
  endgroup
endmodule

// CHECK: moore.crossbin.decl @sel kind<bins> {
// CHECK:   moore.binsof @a_cp intersect [511]
// CHECK:   moore.binsof @b_cp intersect [1]
// CHECK: }

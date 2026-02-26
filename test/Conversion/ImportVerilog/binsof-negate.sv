// RUN: circt-verilog %s --ir-moore | FileCheck %s

module test_binsof_negate;
  logic clk;
  logic [7:0] a;
  logic [7:0] b;

  // CHECK: moore.covercross.decl @c targets [@a_cp, @b_cp] {
  // CHECK:   moore.crossbin.decl @n1 kind<bins> {
  // CHECK:     moore.binsof @a_cp intersect [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200] negate
  // CHECK:   }
  // CHECK:   moore.crossbin.decl @n2 kind<bins> {
  // CHECK:     moore.binsof @a_cp::@a2 negate
  // CHECK:   }
  // CHECK: }
  covergroup cg @(posedge clk);
    a_cp: coverpoint a {
      bins a1 = {[0:63]};
      bins a2 = {[64:127]};
      bins a3 = {[128:191]};
      bins a4 = {[192:255]};
    }
    b_cp: coverpoint b;
    c: cross a_cp, b_cp {
      bins n1 = !binsof(a_cp) intersect {[100:200]};
      bins n2 = !binsof(a_cp.a2);
    }
  endgroup
endmodule

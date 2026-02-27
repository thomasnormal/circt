// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Regression: dynamic 4-state casez/casex matching must honor unknown bits in
// the case expression (not just constant case items).

module top;
  logic [2:0] selx;
  logic [2:0] selz;
  logic [7:0] yx;
  logic [7:0] yz;

  always_comb begin
    casex (selx)
      3'bz1z: yx = 8'h07;
      3'bzx1: yx = 8'hFE;
      3'bx01: yx = 8'h5B;
      3'b111: yx = 8'h00;
      3'b0z0: yx = 8'hEF;
      default: yx = 8'h56;
    endcase
  end

  always_comb begin
    casez (selz)
      3'b100: yz = 8'hD0;
      3'b1x0: yz = 8'hE0;
      default: yz = 8'hF0;
    endcase
  end

  initial begin
    selx = 3'b11z; #1;
    // CHECK: casex0=7
    $display("casex0=%0h", yx);

    selx = 3'bzz1; #1;
    // CHECK: casex1=7
    $display("casex1=%0h", yx);

    selz = 3'b1x0; #1;
    // CHECK: casez_xsig=e0
    $display("casez_xsig=%0h", yz);

    $finish;
  end
endmodule

// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
//
// Unknown 4-state samples should not count toward auto bins. This covers the
// case where value bits are otherwise concrete but unknown-mask is set.

module top;
  logic a, q, c, d;

  covergroup cg;
    cp: coverpoint {a, q, c, d};
  endgroup
  cg cov = new;

  initial begin
    a = 0;
    c = 0;
    d = 0;

    // This sample is unknown and must not create a concrete bin hit.
    q = 1'bx;
    cov.sample();

    // Sample exactly 8 concrete bins ({a,c,d} with q fixed at 1).
    for (int i = 0; i < 8; ++i) begin
      a = i[0];
      c = i[1];
      d = i[2];
      q = 1'b1;
      cov.sample();
    end

    $display("CP_COV=%0.2f", cov.cp.get_coverage());
    $finish;
  end
endmodule

// CHECK: CP_COV=50.00

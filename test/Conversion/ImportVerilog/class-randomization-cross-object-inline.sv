// REQUIRES: circt-sim
// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top=test 2>&1 | FileCheck %s

// Regression: inline cross-object equality constraints must participate in
// constrained randomization and must not be overwritten by unconstrained
// fallback randomization.

class Req;
  rand logic [1:0] hwrite;
endclass

class Seq;
  rand logic [1:0] hwriteSeq;
endclass

module test;
  initial begin
    Req r;
    Seq s;
    int all_ok;
    r = new();
    s = new();
    all_ok = 1;

    repeat (20) begin
      void'(s.randomize() with { hwriteSeq == 2'b01; });
      void'(r.randomize() with { hwrite == s.hwriteSeq; });
      if (r.hwrite != s.hwriteSeq)
        all_ok = 0;
    end

    $display("cross_eq_ok=%0d", all_ok);
  end
endmodule

// CHECK: cross_eq_ok=1

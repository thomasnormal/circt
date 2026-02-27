// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// REQUIRES: slang
//
// Regression: cross sampling must respect iff guards per participating
// coverpoint. A false iff on an unrelated coverpoint must not suppress
// unrelated crosses.

// VERILOG-NOT: error

module top;
  int a, b, c;
  bit en1;

  covergroup cg;
    cp1: coverpoint a iff (en1);
    cp2: coverpoint b;
    cp3: coverpoint c;
    x12: cross cp1, cp2;
    x23: cross cp2, cp3;
  endgroup

  initial begin
    static cg cg_inst = new();

    en1 = 0;
    a = 9;
    b = 1;
    c = 2;
    cg_inst.sample();

    // x12 uses cp1 -> should not sample when cp1 iff is false.
    // CHECK: x12_nonzero=0
    $display("x12_nonzero=%0d", cg_inst.x12.get_inst_coverage() > 0.0);
    // x23 does not use cp1 -> should still sample.
    // CHECK: x23_nonzero=1
    $display("x23_nonzero=%0d", cg_inst.x23.get_inst_coverage() > 0.0);

    $finish;
  end
endmodule

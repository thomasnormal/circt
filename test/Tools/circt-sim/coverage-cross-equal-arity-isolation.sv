// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// REQUIRES: slang
//
// Regression: multiple crosses with the same arity must not share bin hit data.

// VERILOG-NOT: error

module top;
  int a, b, c, d;

  covergroup cg;
    cpa: coverpoint a;
    cpb: coverpoint b;
    cpc: coverpoint c;
    cpd: coverpoint d;
    xab: cross cpa, cpb;
    xcd: cross cpc, cpd;
  endgroup

  initial begin
    static cg cg_inst = new();

    // xab tuples: (0,0), (1,1), (2,2), (3,3)      => 4 bins
    // xcd tuples: (0,3), (1,2), (2,1), (3,0)      => 4 bins
    // For each cross: unique axis values = 4 x 4 = 16 bins total.
    // Expected per-cross coverage: 4/16 = 25%.
    a = 0; b = 0; c = 0; d = 3; cg_inst.sample();
    a = 1; b = 1; c = 1; d = 2; cg_inst.sample();
    a = 2; b = 2; c = 2; d = 1; cg_inst.sample();
    a = 3; b = 3; c = 3; d = 0; cg_inst.sample();

    // CHECK: xab_cov=25
    $display("xab_cov=%0d", $rtoi(cg_inst.xab.get_inst_coverage()));
    // CHECK: xcd_cov=25
    $display("xcd_cov=%0d", $rtoi(cg_inst.xcd.get_inst_coverage()));

    $finish;
  end
endmodule

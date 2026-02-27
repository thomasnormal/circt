// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --mode interpret --top top 2>&1 | FileCheck %s
// REQUIRES: slang
//
// Regression: in interpret mode, named cross-bin filters must be honored.

// VERILOG-NOT: error

module top;
  int a, b;

  covergroup cg;
    cpa: coverpoint a;
    cpb: coverpoint b;
    xab: cross cpa, cpb {
      // Ignore only the (0,0) tuple.
      ignore_bins only_zero_zero =
          binsof(cpa) intersect {0} && binsof(cpb) intersect {0};
    }
  endgroup

  initial begin
    static cg cg_inst = new();

    // One ignored sample, one non-ignored sample.
    a = 0; b = 0; cg_inst.sample();
    a = 1; b = 1; cg_inst.sample();

    // If filters are honored, coverage must be non-zero.
    // CHECK: cross_cov_nonzero=1
    $display("cross_cov_nonzero=%0d", cg_inst.xab.get_inst_coverage() > 0.0);

    $finish;
  end
endmodule

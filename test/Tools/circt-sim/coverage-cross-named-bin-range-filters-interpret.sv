// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --mode interpret --top top 2>&1 | FileCheck %s
// REQUIRES: slang
//
// Regression: named cross-bin filters that use intersect ranges must be
// decoded and honored in interpret mode.

// VERILOG-NOT: error

module top;
  int a, b;

  covergroup cg;
    cpa: coverpoint a;
    cpb: coverpoint b;
    xab: cross cpa, cpb {
      // Ignore only the [0:1]x[0:1] region.
      ignore_bins low_corner =
          binsof(cpa) intersect {[0:1]} && binsof(cpb) intersect {[0:1]};
    }
  endgroup

  initial begin
    static cg cg_inst = new();

    // Ignored sample (inside range filter).
    a = 0; b = 1; cg_inst.sample();

    // Non-ignored sample (outside range filter).
    a = 3; b = 3; cg_inst.sample();

    // If range filters are honored, coverage remains non-zero.
    // CHECK: cross_cov_nonzero=1
    $display("cross_cov_nonzero=%0d", cg_inst.xab.get_inst_coverage() > 0.0);

    $finish;
  end
endmodule

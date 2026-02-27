// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// REQUIRES: slang
//
// Regression: coverpoint option.auto_bin_max must be propagated to runtime
// coverage accounting in interpret mode.

// VERILOG-NOT: error

module top;
  int x;

  covergroup cg;
    cp: coverpoint x {
      option.auto_bin_max = 2;
    }
  endgroup

  initial begin
    static cg cg_inst = new();

    x = 0;
    cg_inst.sample();
    x = 2;
    cg_inst.sample();

    // With auto_bin_max=2 and observed range [0..2], effective range is 2,
    // so two covered values should report 100%.
    // CHECK: cp_inst_cov=100
    $display("cp_inst_cov=%0d", $rtoi(cg_inst.cp.get_inst_coverage()));
    $finish;
  end
endmodule

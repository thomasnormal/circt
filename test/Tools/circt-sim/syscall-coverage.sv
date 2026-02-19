// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// TODO: get_inst_coverage() returns x instead of numeric value — coverage query not wired.
// Test covergroup get_coverage() and get_inst_coverage()
module top;
  int val;

  covergroup cg;
    cp: coverpoint val {
      bins low = {[0:9]};
      bins mid = {[10:19]};
      bins high = {[20:29]};
    }
  endgroup

  initial begin
    cg cg_inst = new();

    // Before any sampling, coverage should be 0
    // CHECK: initial_cov=0
    $display("initial_cov=%0d", $rtoi(cg_inst.get_inst_coverage()));

    // Sample one bin
    val = 5;
    cg_inst.sample();
    // Coverage should increase (1 of 3 bins = ~33%)
    // CHECK: one_bin_positive=1
    $display("one_bin_positive=%0d", cg_inst.get_inst_coverage() > 0.0);

    // Sample all bins
    val = 15;
    cg_inst.sample();
    val = 25;
    cg_inst.sample();
    // All 3 bins hit = 100%
    // CHECK: all_bins=100
    $display("all_bins=%0d", $rtoi(cg_inst.get_inst_coverage()));

    $finish;
  end
endmodule

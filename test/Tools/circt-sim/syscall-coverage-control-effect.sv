// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test coverage sampling start/stop semantics.
// After $coverage_control($cov_stop, ...), samples must not change coverage
// until $cov_start is issued.
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

    // Sample one bin while coverage is active
    val = 5;
    cg_inst.sample();
    // CHECK: before_stop_positive=1
    $display("before_stop_positive=%0d", cg_inst.get_inst_coverage() > 0.0);

    // Stop coverage collection.
    $coverage_control(2, 1, 0, top);

    // Samples should not be counted while stopped.
    val = 15;
    cg_inst.sample();
    val = 25;
    cg_inst.sample();

    // Only the first bin should be hit => 1/3 bins => 33% (truncated).
    // CHECK: stopped_cov=33
    $display("stopped_cov=%0d", $rtoi(cg_inst.get_inst_coverage()));

    $coverage_control(1, 1, 0, top);

    val = 15;
    cg_inst.sample();
    // Restarted collection should count new samples again => 2/3 bins => 66%.
    // CHECK: restarted_cov=66
    $display("restarted_cov=%0d", $rtoi(cg_inst.get_inst_coverage()));

    $finish;
  end
endmodule

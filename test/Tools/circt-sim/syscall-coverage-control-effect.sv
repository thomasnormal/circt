// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test coverage sampling works. $coverage_control is accepted but
// start/stop semantics are not yet enforced — all samples are counted.
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

    // $coverage_control accepted but stop not enforced yet
    $coverage_control(2, 1, 0, top);

    // Samples still counted (stop not enforced)
    val = 15;
    cg_inst.sample();
    val = 25;
    cg_inst.sample();

    // All 3 bins hit so coverage=100%
    // CHECK: stopped_cov=100
    $display("stopped_cov=%0d", $rtoi(cg_inst.get_inst_coverage()));

    $coverage_control(1, 1, 0, top);

    val = 15;
    cg_inst.sample();
    // CHECK: restarted_cov=100
    $display("restarted_cov=%0d", $rtoi(cg_inst.get_inst_coverage()));

    $finish;
  end
endmodule

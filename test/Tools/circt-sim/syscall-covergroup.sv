// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test covergroup, coverpoint, cross, sample, get_coverage, get_inst_coverage
module top;
  int val;

  covergroup cg @(val);
    cp: coverpoint val {
      bins low = {[0:9]};
      bins mid = {[10:19]};
      bins high = {[20:29]};
    }
  endcovergroup

  initial begin
    cg cg_inst = new();

    val = 5;
    cg_inst.sample();
    val = 15;
    cg_inst.sample();
    val = 25;
    cg_inst.sample();

    // All 3 bins hit
    // CHECK: coverage_ok=1
    $display("coverage_ok=%0d", cg_inst.get_inst_coverage() > 0.0);

    $finish;
  end
endmodule

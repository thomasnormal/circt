// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
//
// Regression: coverpoint member methods on a covergroup handle must work.

module top;
  int val;

  covergroup cg;
    cp: coverpoint val {
      bins low = {[0:9]};
      bins high = {[10:19]};
    }
  endgroup

  initial begin
    static cg cg_inst = new();

    // CHECK: cp_inst_cov0=0
    $display("cp_inst_cov0=%0d", $rtoi(cg_inst.cp.get_inst_coverage()));
    // CHECK: cp_cov0=0
    $display("cp_cov0=%0d", $rtoi(cg_inst.cp.get_coverage()));

    val = 5;
    cg_inst.sample();
    // CHECK: cp_inst_cov1=50
    $display("cp_inst_cov1=%0d", $rtoi(cg_inst.cp.get_inst_coverage()));
    // CHECK: cp_cov1=50
    $display("cp_cov1=%0d", $rtoi(cg_inst.cp.get_coverage()));

    val = 15;
    cg_inst.sample();
    // CHECK: cp_inst_cov2=100
    $display("cp_inst_cov2=%0d", $rtoi(cg_inst.cp.get_inst_coverage()));
    // CHECK: cp_cov2=100
    $display("cp_cov2=%0d", $rtoi(cg_inst.cp.get_coverage()));

    $finish;
  end
endmodule

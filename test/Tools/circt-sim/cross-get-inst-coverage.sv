// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
//
// Regression: cross member methods on a covergroup handle must work.

module top;
  int a, b;

  covergroup cg;
    cpa: coverpoint a { bins low = {[0:1]}; bins high = {[2:3]}; }
    cpb: coverpoint b { bins low = {[0:1]}; bins high = {[2:3]}; }
    xab: cross cpa, cpb;
  endgroup

  initial begin
    static cg cg_inst = new();

    // CHECK: x_inst_cov0=0
    $display("x_inst_cov0=%0d", $rtoi(cg_inst.xab.get_inst_coverage()));
    // CHECK: x_cov0=0
    $display("x_cov0=%0d", $rtoi(cg_inst.xab.get_coverage()));

    a = 0;
    b = 0;
    cg_inst.sample();
    // CHECK: x_inst_cov1=100
    $display("x_inst_cov1=%0d", $rtoi(cg_inst.xab.get_inst_coverage()));
    // CHECK: x_cov1=100
    $display("x_cov1=%0d", $rtoi(cg_inst.xab.get_coverage()));

    a = 2;
    b = 2;
    cg_inst.sample();
    // CHECK: x_inst_cov2=50
    $display("x_inst_cov2=%0d", $rtoi(cg_inst.xab.get_inst_coverage()));
    // CHECK: x_cov2=50
    $display("x_cov2=%0d", $rtoi(cg_inst.xab.get_coverage()));

    $finish;
  end
endmodule

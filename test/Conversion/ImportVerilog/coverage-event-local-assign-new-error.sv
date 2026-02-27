// RUN: circt-verilog %s --ir-moore --no-uvm-auto-include | FileCheck %s
// REQUIRES: slang

module top;
  logic clk = 0;
  always #5 clk = ~clk;
  logic [3:0] addr = 0;

  covergroup cg @(posedge clk);
    cp_addr: coverpoint addr;
  endgroup

  initial begin
    cg cov;
    cov = new;
    @(posedge clk);
    addr <= addr + 1;
  end
endmodule

// CHECK-LABEL: moore.module @top
// CHECK: %[[COV:.*]] = moore.variable : <covergroup<@cg>>
// CHECK: moore.procedure initial {
// CHECK: %[[NEW:.*]] = moore.covergroup.inst @cg : <@cg>
// CHECK: moore.blocking_assign %[[COV]], %[[NEW]] : covergroup<@cg>
// CHECK: moore.procedure always {
// CHECK: moore.wait_event
// CHECK: moore.detect_event posedge
// CHECK: %[[CGH:.*]] = moore.read %[[COV]] : <covergroup<@cg>>
// CHECK: moore.covergroup.sample %[[CGH]]

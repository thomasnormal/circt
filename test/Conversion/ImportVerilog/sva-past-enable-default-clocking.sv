// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaPastEnableDefaultClocking(
    input logic clk,
    input logic en,
    input logic d,
    output logic q);
  default clocking cb @(posedge clk); endclocking

  // CHECK: moore.procedure always
  // CHECK: moore.detect_event posedge
  // CHECK: moore.conditional
  // CHECK: moore.blocking_assign
  // CHECK: moore.assign
  always_comb q = $past(d, 1, en);
endmodule

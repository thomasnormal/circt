// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledGclkProcedural(
    input logic clk,
    input logic d,
    output logic q);
  global clocking cb @(posedge clk); endclocking

  // `_gclk` sampled functions should sample on global clock in procedural use.
  // CHECK: moore.procedure always
  // CHECK: moore.detect_event posedge
  // CHECK: moore.not
  // CHECK: moore.assign
  always_comb q = $changed_gclk(d);
endmodule

// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledDefaultClockingProcedural(
    input logic clk,
    input logic d,
    output logic q);
  default clocking cb @(posedge clk); endclocking

  // Procedural sampled-value calls should use inferred default clocking.
  // CHECK: moore.procedure always
  // CHECK: moore.detect_event posedge
  // CHECK: moore.not
  // CHECK: moore.assign
  always_comb q = $changed(d);
endmodule

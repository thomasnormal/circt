// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module SvaPastCovergroupSampledControls(input logic clk, input logic a);
  covergroup cg_t;
    coverpoint a;
  endgroup

  cg_t cg;
  initial cg = new();

  property p_cg_past;
    @(posedge clk) $past(cg, 1, 1'b1, @(posedge clk)) == cg;
  endproperty
  assert property (p_cg_past);

  property p_cg_past_disable;
    disable iff (!a) @(posedge clk) $past(cg, 2, 1'b1, @(posedge clk)) != cg;
  endproperty
  cover property (p_cg_past_disable);
endmodule

// CHECK-LABEL: moore.module @SvaPastCovergroupSampledControls
// CHECK: moore.wait_event
// CHECK: moore.detect_event posedge
// CHECK: moore.covergroup_handle_cmp
// CHECK: verif.{{(clocked_)?}}assert
// CHECK: verif.{{(clocked_)?}}cover

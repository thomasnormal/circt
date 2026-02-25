// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledUnpackedStruct(input logic clk_a, input logic clk_b);
  typedef struct {
    logic [1:0] a;
    logic b;
  } s_t;
  s_t s;

  // Regular assertion-clock sampled struct support.
  // CHECK: moore.past
  // CHECK: moore.struct_extract
  // CHECK: moore.case_eq
  // CHECK: moore.and
  // CHECK: moore.not
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk_a) $changed(s));

  // Explicit sampled clock should also work (helper path).
  // CHECK: moore.procedure always
  // CHECK: moore.struct_extract
  // CHECK: moore.case_eq
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk_a) $stable(s, @(posedge clk_b)));
endmodule

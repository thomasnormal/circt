// RUN: circt-verilog %s --ir-hw | FileCheck %s
// REQUIRES: slang

module top(
    input logic [3:0] idx,
    input logic in,
    output logic out
);
  typedef struct packed {
    logic [6:0] lru;
    logic [7:0] dirty;
  } dirty_stat_s;

  dirty_stat_s dirty_stat_r;

  assign out = dirty_stat_r[idx];

  always_comb begin
    dirty_stat_r[idx] = in;
  end
endmodule

// CHECK-LABEL: hw.module @top
// CHECK: builtin.unrealized_conversion_cast %dirty_stat_r
// CHECK-SAME: to !llhd.ref<!hw.struct<value: i15, unknown: i15>>
// CHECK: llhd.drv

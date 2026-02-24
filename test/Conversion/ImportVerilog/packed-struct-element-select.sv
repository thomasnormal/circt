// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module PackedStructElementSelect;
  typedef struct packed {
    logic [2:0] flags;
    logic dirty;
  } cache_stat_t;

  cache_stat_t stat;
  logic [1:0] idx;
  logic rd;
  logic wr;

  // CHECK: [[STAT_READ:%.+]] = moore.read %stat : <struct<{flags: l3, dirty: l1}>>
  // CHECK: [[IDX_READ:%.+]] = moore.read %idx : <l2>
  // CHECK: [[EXT:%.+]] = moore.dyn_extract [[STAT_READ]] from [[IDX_READ]] : struct<{flags: l3, dirty: l1}>, l2 -> l1
  // CHECK: moore.assign %rd, [[EXT]] : l1
  assign rd = stat[idx];

  always_comb begin
    // CHECK: [[IDX_LHS:%.+]] = moore.read %idx : <l2>
    // CHECK: [[LHS_REF:%.+]] = moore.dyn_extract_ref %stat from [[IDX_LHS]] : <struct<{flags: l3, dirty: l1}>>, l2 -> <l1>
    // CHECK: [[WR_READ:%.+]] = moore.read %wr : <l1>
    // CHECK: moore.blocking_assign [[LHS_REF]], [[WR_READ]] : l1
    stat[idx] = wr;
  end
endmodule

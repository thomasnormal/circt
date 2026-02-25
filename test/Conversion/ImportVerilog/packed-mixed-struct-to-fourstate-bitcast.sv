// RUN: circt-verilog %s --ir-hw | FileCheck %s
// REQUIRES: slang

module top;
  typedef struct packed {
    logic [38:0] vaddr;
    bit   [31:0] op;
    logic        spec;
  } fetch_t;

  fetch_t f;
  logic [71:0] bits;

  always_comb bits = f;
endmodule

// CHECK-LABEL: hw.module @top
// CHECK: comb.concat {{.*}} : i39, i32, i1
// CHECK: comb.concat {{.*}}%c0_i32{{.*}} : i39, i32, i1
// CHECK: hw.struct_create

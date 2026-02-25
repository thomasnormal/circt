// RUN: circt-verilog %s --ir-hw | FileCheck %s
// REQUIRES: slang

module top;
  typedef struct packed {
    logic a;
    bit   b;
  } elem_t;

  elem_t [1:0] arr;
  logic [3:0] bits;

  always_comb bits = arr;
endmodule

// CHECK-LABEL: hw.module @top
// CHECK: hw.array_get
// CHECK: comb.concat {{.*}} : i1, i1, i1, i1
// CHECK: comb.concat {{.*}}, %false, {{.*}}, %false : i1, i1, i1, i1
// CHECK: hw.struct_create

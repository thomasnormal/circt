// RUN: circt-verilog %s --ir-hw | FileCheck %s
// REQUIRES: slang

module top;
  typedef union packed {
    logic [3:0] req;
    logic [3:0] cmd;
    logic [3:0] fill;
    logic [3:0] resp;
  } msg_t;

  typedef struct packed {
    logic [39:0] addr;
    msg_t msg_type;
  } header_t;

  header_t h;
  logic [43:0] bits;

  always_comb bits = h;
endmodule

// CHECK-LABEL: hw.module @top
// CHECK: msg_type: !hw.union<req: !hw.struct<value: i4, unknown: i4>
// CHECK: hw.bitcast

module top_direct;
  typedef union packed {
    logic [3:0] req;
    logic [3:0] cmd;
  } msg_t;

  logic [3:0] x;
  msg_t u;

  always_comb u = x;
endmodule

// CHECK-LABEL: hw.module @top_direct
// CHECK: hw.bitcast {{.*}} : (!hw.struct<value: i4, unknown: i4>) -> !hw.union<req: !hw.struct<value: i4, unknown: i4>, cmd: !hw.struct<value: i4, unknown: i4>>

// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_abort_on(input logic clk, rst_n, a, b, c);
  // CHECK-LABEL: moore.module @sva_abort_on

  // CHECK-DAG: ltl.or
  assert property (@(posedge clk) accept_on(!rst_n) (a |-> b));

  // CHECK-DAG: ltl.and
  // CHECK-DAG: ltl.not
  assert property (@(posedge clk) reject_on(!rst_n) (a |-> b));

  // CHECK-DAG: ltl.or
  assert property (@(posedge clk) sync_accept_on(c) (a |=> b));

  // CHECK-DAG: ltl.and
  // CHECK-DAG: ltl.not
  assert property (@(posedge clk) sync_reject_on(c) (a |=> b));
endmodule

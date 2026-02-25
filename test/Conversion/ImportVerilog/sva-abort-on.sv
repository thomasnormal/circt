// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_abort_on(input logic clk, rst_n, a, b, c);
  // CHECK-LABEL: moore.module @sva_abort_on

  // Async accept_on: condition is used directly, then property is clocked.
  // CHECK: [[A_COND:%.*]] = moore.to_builtin_bool
  // CHECK: [[A_IMPL:%.*]] = ltl.implication
  // CHECK: [[A_OR:%.*]] = ltl.or [[A_COND]], [[A_IMPL]]
  // CHECK: verif.clocked_assert [[A_OR]], posedge
  assert property (@(posedge clk) accept_on(!rst_n) (a |-> b));

  // Async reject_on: negated condition is used directly, then property is clocked.
  // CHECK: [[R_NOT:%.*]] = ltl.not [[A_COND]]
  // CHECK: [[R_AND:%.*]] = ltl.and [[R_NOT]], [[A_IMPL]]
  // CHECK: verif.clocked_assert [[R_AND]], posedge
  assert property (@(posedge clk) reject_on(!rst_n) (a |-> b));

  // Sync accept_on: condition is sampled on the property clock before OR.
  // CHECK: [[SACOND:%.*]] = moore.to_builtin_bool
  // CHECK: [[SACONDCLK:%.*]] = ltl.clock [[SACOND]],  posedge
  // CHECK: [[SAIMPL:%.*]] = ltl.implication
  // CHECK: [[SAOR:%.*]] = ltl.or [[SACONDCLK]], [[SAIMPL]]
  // CHECK: verif.clocked_assert [[SAOR]], posedge
  assert property (@(posedge clk) sync_accept_on(c) (a |=> b));

  // Sync reject_on: sampled condition is negated and ANDed with property.
  // CHECK: [[SRNOT:%.*]] = ltl.not [[SACONDCLK]]
  // CHECK: [[SRAND:%.*]] = ltl.and [[SRNOT]], [[SAIMPL]]
  // CHECK: verif.clocked_assert [[SRAND]], posedge
  assert property (@(posedge clk) sync_reject_on(c) (a |=> b));
endmodule

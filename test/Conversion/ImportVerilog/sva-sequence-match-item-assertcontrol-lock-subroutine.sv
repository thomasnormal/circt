// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=DIAG
// REQUIRES: slang

module SVAAssertControlLockMatchItem(input logic clk, a);
  sequence s_lock;
    (1, $assertcontrol(1), $assertoff(), $assertcontrol(2), $assertoff()) ##1 a;
  endsequence

  assert property (@(posedge clk) s_lock);

  // CHECK-LABEL: moore.global_variable @__circt_assert_control_locked
  // CHECK-LABEL: moore.module @SVAAssertControlLockMatchItem
  // CHECK-DAG: moore.constant 1 : i32
  // CHECK-DAG: moore.constant 2 : i32
  // CHECK: moore.get_global_variable @__circt_assert_control_locked : <i1>
  // CHECK: moore.get_global_variable @__circt_proc_assertions_enabled : <i1>
  // CHECK: moore.conditional
  // CHECK: moore.blocking_assign {{%.*}}, {{%.*}} : i1

  // DIAG-NOT: ignoring system subroutine `$assertcontrol`
  // DIAG-NOT: ignoring system subroutine `$assertoff`
endmodule

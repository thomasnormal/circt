// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-translate --import-verilog %s 2>&1 | FileCheck %s --check-prefix=DIAG
// REQUIRES: slang

module SVAAssertControlLockMatchItem(input logic clk, a);
  sequence s_lock;
    (1, $assertcontrol(1), $assertoff(), $assertcontrol(2), $assertoff()) ##1 a;
  endsequence

  assert property (@(posedge clk) s_lock);

  // CHECK-LABEL: moore.global_variable @__circt_assert_control_locked
  // CHECK-LABEL: moore.module @SVAAssertControlLockMatchItem
  // CHECK: %[[CTYPE_LOCK:.*]] = moore.constant 1 : i32
  // CHECK: %[[LOCKREF0:.*]] = moore.get_global_variable @__circt_assert_control_locked : <i1>
  // CHECK: %[[LOCKCUR0:.*]] = moore.read %[[LOCKREF0]] : <i1>
  // CHECK: moore.blocking_assign %{{.*}}, %{{.*}} : i1
  // CHECK: %[[LOCKREF1:.*]] = moore.get_global_variable @__circt_assert_control_locked : <i1>
  // CHECK: %[[LOCKCUR1:.*]] = moore.read %[[LOCKREF1]] : <i1>
  // CHECK: %[[PROCREF1:.*]] = moore.get_global_variable @__circt_proc_assertions_enabled : <i1>
  // CHECK: %[[PROCCUR1:.*]] = moore.read %[[PROCREF1]] : <i1>
  // CHECK: %[[PROCNEXT1:.*]] = moore.conditional %{{.*}} : i1 -> i1
  // CHECK: moore.blocking_assign %{{.*}}, %[[PROCNEXT1]] : i1
  // CHECK: %[[CTYPE_UNLOCK:.*]] = moore.constant 2 : i32
  // CHECK: moore.blocking_assign %{{.*}}, %{{.*}} : i1

  // DIAG-NOT: ignoring system subroutine `$assertcontrol`
  // DIAG-NOT: ignoring system subroutine `$assertoff`
endmodule

// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module SVAAssertControlLockProcedural(input logic a);
  initial begin
    // Lock assertion control updates.
    $assertcontrol(1);
    // While locked, this should not directly force the procedural enable off.
    $assertoff;

    // Unlock and then apply an update.
    $assertcontrol(2);
    $assertoff;
    assert (a) else $error("x");
  end

  // CHECK-LABEL: moore.global_variable @__circt_assert_control_locked
  // CHECK-LABEL: moore.module @SVAAssertControlLockProcedural
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
endmodule

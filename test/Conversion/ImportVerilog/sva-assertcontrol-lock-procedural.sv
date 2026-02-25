// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
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
  // CHECK-DAG: moore.constant 1 : i32
  // CHECK-DAG: moore.constant 2 : i32
  // CHECK: moore.get_global_variable @__circt_assert_control_locked : <i1>
  // CHECK: moore.get_global_variable @__circt_proc_assertions_enabled : <i1>
  // CHECK: moore.conditional
  // CHECK: moore.blocking_assign {{%.*}}, {{%.*}} : i1
endmodule

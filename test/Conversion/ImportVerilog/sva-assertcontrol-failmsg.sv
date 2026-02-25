// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module SVAAssertControlFailMessages(input logic a);
  initial begin
    // $assertcontrol(9) should map to fail-message off.
    $assertcontrol(9);
    assert (a) else $error("suppressed");

    // $assertcontrol(8) should map to fail-message on.
    $assertcontrol(8);
    assert (a) else $error("enabled");
  end

  // CHECK-LABEL: moore.global_variable @__circt_assert_fail_msgs_enabled
  // CHECK-LABEL: moore.module @SVAAssertControlFailMessages
  // CHECK-DAG: moore.constant 9 : i32
  // CHECK-DAG: moore.constant 8 : i32
  // CHECK: moore.get_global_variable @__circt_assert_fail_msgs_enabled : <i1>
  // CHECK: moore.eq {{%.*}}, {{%.*}} : i32 -> i1
  // CHECK: moore.eq {{%.*}}, {{%.*}} : i32 -> i1
  // CHECK: moore.conditional
  // CHECK: moore.blocking_assign {{%.*}}, {{%.*}} : i1
endmodule

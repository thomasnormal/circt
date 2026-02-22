// RUN: circt-translate --import-verilog %s | FileCheck %s
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
  // CHECK: %[[CTYPE9:.*]] = moore.constant 9 : i32
  // CHECK: %[[FAILCTL1:.*]] = moore.get_global_variable @__circt_assert_fail_msgs_enabled : <i1>
  // CHECK: %[[FAILCUR1:.*]] = moore.read %[[FAILCTL1]] : <i1>
  // CHECK: %[[C8A:.*]] = moore.constant 8 : i32
  // CHECK: %[[C9A:.*]] = moore.constant 9 : i32
  // CHECK: %[[IS8A:.*]] = moore.eq %[[CTYPE9]], %[[C8A]] : i32 -> i1
  // CHECK: %[[IS9A:.*]] = moore.eq %[[CTYPE9]], %[[C9A]] : i32 -> i1
  // CHECK: %[[FAILAFTER9:.*]] = moore.conditional %[[IS9A]] : i1 -> i1
  // CHECK: %[[FAILNEXT9:.*]] = moore.conditional %[[IS8A]] : i1 -> i1
  // CHECK: %[[FAILCTL2:.*]] = moore.get_global_variable @__circt_assert_fail_msgs_enabled : <i1>
  // CHECK: moore.blocking_assign %[[FAILCTL2]], %[[FAILNEXT9]] : i1
  // CHECK: %[[CTYPE8:.*]] = moore.constant 8 : i32
  // CHECK: %[[IS8B:.*]] = moore.eq %[[CTYPE8]], {{%.*}} : i32 -> i1
  // CHECK: %[[IS9B:.*]] = moore.eq %[[CTYPE8]], {{%.*}} : i32 -> i1
endmodule

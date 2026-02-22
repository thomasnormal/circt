// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-translate --import-verilog %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemAssertControlSubroutine(input logic clk, a);
  sequence s_off;
    (1, $assertoff()) ##1 a;
  endsequence
  sequence s_on;
    (1, $asserton()) ##1 a;
  endsequence
  sequence s_control;
    (1, $assertcontrol(3), $assertcontrol(4), $assertcontrol(5),
     $assertcontrol(6), $assertcontrol(7), $assertcontrol(8),
     $assertcontrol(9), $assertcontrol(10), $assertcontrol(11)) ##1 a;
  endsequence
  sequence s_fail;
    (1, $assertfailoff(), $assertfailon()) ##1 a;
  endsequence
  sequence s_other;
    (1, $assertpasson(), $assertpassoff(), $assertnonvacuouson(),
     $assertvacuousoff()) ##1 a;
  endsequence

  // Assertion-control match-item tasks should preserve control side effects.
  // CHECK-DAG: moore.global_variable @__circt_proc_assertions_enabled
  // CHECK-DAG: moore.global_variable @__circt_assert_fail_msgs_enabled
  // CHECK-DAG: moore.global_variable @__circt_assert_pass_msgs_enabled
  // CHECK-DAG: moore.global_variable @__circt_assert_vacuous_pass_enabled
  // CHECK-LABEL: moore.module @SVASequenceMatchItemAssertControlSubroutine
  // CHECK: moore.constant 10 : i32
  // CHECK: moore.constant 11 : i32
  // CHECK: moore.blocking_assign
  // CHECK: verif.assert
  assert property (@(posedge clk) s_off);
  assert property (@(posedge clk) s_on);
  assert property (@(posedge clk) s_control);
  assert property (@(posedge clk) s_fail);
  assert property (@(posedge clk) s_other);
endmodule

// DIAG-NOT: ignoring system subroutine `$assertoff`
// DIAG-NOT: ignoring system subroutine `$asserton`
// DIAG-NOT: ignoring system subroutine `$assertkill`
// DIAG-NOT: ignoring system subroutine `$assertcontrol`
// DIAG-NOT: ignoring system subroutine `$assertfailoff`
// DIAG-NOT: ignoring system subroutine `$assertfailon`
// DIAG-NOT: ignoring system subroutine `$assertpasson`
// DIAG-NOT: ignoring system subroutine `$assertpassoff`
// DIAG-NOT: ignoring system subroutine `$assertnonvacuouson`
// DIAG-NOT: ignoring system subroutine `$assertvacuousoff`

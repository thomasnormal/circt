// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemAssertControlPassVacuousSubroutine(input logic clk, a);
  sequence s_control_pass_vacuous;
    (1, $assertcontrol(6), $assertcontrol(7),
     $assertcontrol(10), $assertcontrol(11)) ##1 a;
  endsequence

  // CHECK-DAG: moore.global_variable @__circt_assert_pass_msgs_enabled
  // CHECK-DAG: moore.global_variable @__circt_assert_vacuous_pass_enabled
  // CHECK-LABEL: moore.module @SVASequenceMatchItemAssertControlPassVacuousSubroutine
  // CHECK: moore.get_global_variable @__circt_assert_pass_msgs_enabled
  // CHECK: moore.blocking_assign
  // CHECK: moore.get_global_variable @__circt_assert_vacuous_pass_enabled
  // CHECK: moore.blocking_assign
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) s_control_pass_vacuous);
endmodule

// DIAG-NOT: ignoring system subroutine `$assertcontrol`

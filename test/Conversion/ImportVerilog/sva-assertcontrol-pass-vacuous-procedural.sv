// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module SVAAssertControlPassOn(input logic a);
  initial begin
    $assertcontrol(6);
    assert (a) else $error("check");
  end

  // CHECK-LABEL: moore.module @SVAAssertControlPassOn
  // CHECK: moore.constant 6 : i32
  // CHECK: moore.constant 7 : i32
  // CHECK: moore.get_global_variable @__circt_assert_pass_msgs_enabled : <i1>
  // CHECK: moore.blocking_assign %{{.*}}, %{{.*}} : i1
endmodule

module SVAAssertControlNonVacuousOn(input logic a);
  initial begin
    $assertcontrol(10);
    assert (a) else $error("check");
  end

  // CHECK-LABEL: moore.module @SVAAssertControlNonVacuousOn
  // CHECK: moore.constant 10 : i32
  // CHECK: moore.constant 11 : i32
  // CHECK: moore.get_global_variable @__circt_assert_vacuous_pass_enabled : <i1>
  // CHECK: moore.blocking_assign %{{.*}}, %{{.*}} : i1
endmodule

module SVAAssertPassVacuousSubroutines(input logic a);
  initial begin
    $assertpasson;
    $assertpassoff;
    $assertnonvacuouson;
    $assertvacuousoff;
    assert (a) else $error("check");
  end

  // CHECK-LABEL: moore.module @SVAAssertPassVacuousSubroutines
  // CHECK-DAG: moore.get_global_variable @__circt_assert_control_locked : <i1>
  // CHECK-DAG: moore.constant 1 : i1
  // CHECK-DAG: moore.constant 0 : i1
  // CHECK-DAG: moore.get_global_variable @__circt_assert_pass_msgs_enabled : <i1>
  // CHECK-DAG: moore.get_global_variable @__circt_assert_vacuous_pass_enabled : <i1>
  // CHECK: moore.blocking_assign %{{.*}}, %{{.*}} : i1
endmodule

// RUN: circt-opt %s | circt-opt | FileCheck %s

// Test basic coverage operations

hw.module @TestModule(in %clk: !seq.clock, in %x: i4, in %y: i4, in %cond: i1, in %state: i3, out u: i4) {
  // CHECK-LABEL: hw.module @TestModule

  // Test line coverage
  // CHECK: coverage.line "test.sv" line 10
  coverage.line "test.sv" line 10

  // Test line coverage with tag
  // CHECK: coverage.line "test.sv" line 20 tag "block1"
  coverage.line "test.sv" line 20 tag "block1"

  // Test toggle coverage
  // CHECK: coverage.toggle %x name "input_x"
  coverage.toggle %x name "input_x" : i4

  // Test toggle coverage with hierarchy
  // CHECK: coverage.toggle %y name "input_y" hierarchy "top.dut"
  coverage.toggle %y name "input_y" hierarchy "top.dut" : i4

  // Test branch coverage
  // CHECK: coverage.branch %cond name "branch_0" true_id 0 false_id 1
  coverage.branch %cond name "branch_0" true_id 0 false_id 1 : i1

  // Test branch coverage with location
  // CHECK: coverage.branch %cond name "branch_1" true_id 2 false_id 3 at "test.sv" : 42
  coverage.branch %cond name "branch_1" true_id 2 false_id 3 at "test.sv" : 42 : i1

  // Test FSM state coverage
  // CHECK: coverage.fsm.state %state name "fsm_controller" num_states 8
  coverage.fsm.state %state name "fsm_controller" num_states 8 : i3

  // Test FSM state coverage with hierarchy and state names
  // CHECK: coverage.fsm.state %state name "protocol_fsm" num_states 4 hierarchy "top.fsm"
  coverage.fsm.state %state name "protocol_fsm" num_states 4 hierarchy "top.fsm" state_names ["IDLE", "INIT", "RUN", "DONE"] : i3

  // Test FSM transition coverage
  // CHECK: coverage.fsm.transition %state, %state name "fsm_trans" num_states 8
  coverage.fsm.transition %state, %state name "fsm_trans" num_states 8 : i3, i3

  // Test expression coverage
  // CHECK: coverage.expression %cond, %cond name "expr_and"
  coverage.expression %cond, %cond name "expr_and" : i1, i1

  // Test expression coverage with names
  // CHECK: coverage.expression %cond, %cond name "expr_or" hierarchy "top" condition_names ["a", "b"]
  coverage.expression %cond, %cond name "expr_or" hierarchy "top" condition_names ["a", "b"] : i1, i1

  // Test assertion coverage
  // CHECK: coverage.assertion %cond name "valid_check"
  coverage.assertion %cond name "valid_check" : i1

  // Test assertion coverage with location
  // CHECK: coverage.assertion %cond name "valid_check2" hierarchy "top.dut" at "test.sv" : 100
  coverage.assertion %cond name "valid_check2" hierarchy "top.dut" at "test.sv" : 100 : i1

  %0 = comb.add %x, %y : i4
  hw.output %0 : i4
}

// Test coverage group
// CHECK-LABEL: hw.module @TestCoverageGroup
hw.module @TestCoverageGroup(in %x: i4, out y: i4) {
  // CHECK: coverage.group "module_coverage"
  coverage.group "module_coverage" {
    coverage.line "test.sv" line 50
    coverage.toggle %x name "grouped_signal" : i4
  }

  // CHECK: coverage.group "detailed_coverage" description "Coverage for critical path"
  coverage.group "detailed_coverage" description "Coverage for critical path" {
    coverage.line "test.sv" line 60
  }

  hw.output %x : i4
}

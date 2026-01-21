// RUN: circt-opt %s | circt-opt | FileCheck %s

// Test basic coverage operations

// CHECK-LABEL: func.func @test_line_coverage
func.func @test_line_coverage() {
  // CHECK: coverage.line "test.sv" line 10
  coverage.line "test.sv" line 10
  // CHECK: coverage.line "module.sv" line 42 tag "state_machine"
  coverage.line "module.sv" line 42 tag "state_machine"
  return
}

// CHECK-LABEL: func.func @test_toggle_coverage
func.func @test_toggle_coverage(%signal: i8, %clock: i1) {
  // CHECK: coverage.toggle %arg0 name "data_in" : i8
  coverage.toggle %signal name "data_in" : i8
  // CHECK: coverage.toggle %arg1 name "clk_enable" hierarchy "top.sub" : i1
  coverage.toggle %clock name "clk_enable" hierarchy "top.sub" : i1
  return
}

// CHECK-LABEL: func.func @test_branch_coverage
func.func @test_branch_coverage(%cond: i1) {
  // CHECK: coverage.branch %arg0 name "fsm_cond" true_id 0 false_id 1 : i1
  coverage.branch %cond name "fsm_cond" true_id 0 false_id 1 : i1
  // CHECK: coverage.branch %arg0 name "if_stmt" true_id 2 false_id 3 at "test.sv" : 25 : i1
  coverage.branch %cond name "if_stmt" true_id 2 false_id 3 at "test.sv" : 25 : i1
  return
}

// CHECK-LABEL: func.func @test_coverage_group
func.func @test_coverage_group(%signal: i4) {
  // CHECK: coverage.group "fsm_coverage"
  coverage.group "fsm_coverage" {
    // CHECK: coverage.line "fsm.sv" line 10
    coverage.line "fsm.sv" line 10
    // CHECK: coverage.toggle %arg0 name "state" : i4
    coverage.toggle %signal name "state" : i4
  }
  // CHECK: coverage.group "alu_coverage" description "ALU coverage points"
  coverage.group "alu_coverage" description "ALU coverage points" {
    coverage.line "alu.sv" line 20
  }
  return
}

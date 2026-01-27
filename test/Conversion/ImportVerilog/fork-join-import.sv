// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test fork/join statement import into Moore IR.
// fork/join creates parallel execution branches where:
// - join: blocks until all branches complete
// - join_any: blocks until any one branch completes
// - join_none: does not block, spawns branches and continues immediately

// CHECK-LABEL: moore.module @ForkJoinBasic
module ForkJoinBasic;
  initial begin
    // Test fork...join_none (non-blocking - parent continues immediately)
    // CHECK: moore.fork join_none {
    // CHECK:   moore.wait_delay
    // CHECK: }, {
    // CHECK:   moore.wait_delay
    // CHECK: }
    fork
      begin #10; end
      begin #20; end
    join_none

    // Test fork...join (blocking - wait for all branches)
    // CHECK: moore.fork {
    // CHECK:   moore.wait_delay
    // CHECK: }, {
    // CHECK:   moore.wait_delay
    // CHECK: }
    fork
      begin #30; end
      begin #40; end
    join

    // Test fork...join_any (blocking - wait for any branch)
    // CHECK: moore.fork join_any {
    // CHECK:   moore.wait_delay
    // CHECK: }, {
    // CHECK:   moore.wait_delay
    // CHECK: }
    fork
      begin #50; end
      begin #60; end
    join_any
  end
endmodule

// CHECK-LABEL: moore.module @ForkJoinMultipleBranches
module ForkJoinMultipleBranches;
  initial begin
    // Test fork with more than two branches
    // CHECK: moore.fork join_none {
    // CHECK:   moore.wait_delay
    // CHECK: }, {
    // CHECK:   moore.wait_delay
    // CHECK: }, {
    // CHECK:   moore.wait_delay
    // CHECK: }
    fork
      begin #10; end
      begin #20; end
      begin #30; end
    join_none
  end
endmodule

// CHECK-LABEL: moore.module @ForkJoinSingleBranch
module ForkJoinSingleBranch;
  initial begin
    // Test fork with a single branch (edge case)
    // CHECK: moore.fork {
    // CHECK:   moore.wait_delay
    // CHECK: }
    fork
      begin #10; end
    join
  end
endmodule

// CHECK-LABEL: moore.module @ForkJoinNested
module ForkJoinNested;
  initial begin
    // Test nested fork statements
    // CHECK: moore.fork {
    // CHECK:   moore.fork join_none {
    // CHECK:     moore.wait_delay
    // CHECK:   }, {
    // CHECK:     moore.wait_delay
    // CHECK:   }
    // CHECK: }, {
    // CHECK:   moore.wait_delay
    // CHECK: }
    fork
      begin
        fork
          begin #10; end
          begin #20; end
        join_none
      end
      begin #30; end
    join
  end
endmodule

// CHECK-LABEL: moore.module @ForkJoinWithStatements
module ForkJoinWithStatements;
  int a, b, c;

  initial begin
    // Test fork with assignment statements in branches
    // CHECK: moore.fork join_any {
    // CHECK:   moore.blocking_assign
    // CHECK:   moore.wait_delay
    // CHECK: }, {
    // CHECK:   moore.blocking_assign
    // CHECK:   moore.wait_delay
    // CHECK: }
    fork
      begin
        a = 1;
        #10;
      end
      begin
        b = 2;
        #20;
      end
    join_any
  end
endmodule

// CHECK-LABEL: moore.module @ForkJoinEmpty
module ForkJoinEmpty;
  initial begin
    // Test fork with empty branches (edge case)
    // CHECK: moore.fork join_none {
    // CHECK: }, {
    // CHECK: }
    fork
      begin end
      begin end
    join_none
  end
endmodule

// CHECK-LABEL: moore.module @ForkJoinStatementOnly
module ForkJoinStatementOnly;
  int x;
  initial begin
    // Test fork with single statements (no begin/end blocks)
    // CHECK: moore.fork {
    // CHECK:   moore.blocking_assign
    // CHECK: }, {
    // CHECK:   moore.blocking_assign
    // CHECK: }
    fork
      x = 1;
      x = 2;
    join
  end
endmodule

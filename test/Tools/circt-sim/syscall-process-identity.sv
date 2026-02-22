// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test process::self() handle identity:
// - Same process returns same handle on repeated calls
// - Different forked processes return different handles
module top;
  process p1, p2, p3;
  process fork_p;

  initial begin
    p1 = process::self();
    p2 = process::self();

    // Same process should return same handle
    // CHECK: same_handle=1
    $display("same_handle=%0d", p1 == p2);

    // Handle should be non-null
    // CHECK: non_null=1
    $display("non_null=%0d", p1 != null);

    // Fork a child — it should get a DIFFERENT handle
    fork
      begin
        fork_p = process::self();
      end
    join

    // CHECK: different_from_fork=1
    $display("different_from_fork=%0d", p1 != fork_p);

    // Fork handle should also be non-null
    // CHECK: fork_non_null=1
    $display("fork_non_null=%0d", fork_p != null);

    $finish;
  end
endmodule

// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $coverage_control, $coverage_get_max return meaningful values.
// Bug: All coverage system tasks return 0 unconditionally.
// IEEE 1800-2017 Section 20.13:
// - $coverage_control should return 1 on success, 0 on failure
// - $coverage_get_max should return non-zero max coverage value
module top;
  integer ret_start, ret_stop, ret_max;

  initial begin
    // Start coverage — should return 1 (success)
    ret_start = $coverage_control(1, 1, 0, top);
    // CHECK: coverage_start=1
    $display("coverage_start=%0d", ret_start);

    // Stop coverage — should also return 1 (success)
    ret_stop = $coverage_control(2, 1, 0, top);
    // CHECK: coverage_stop=1
    $display("coverage_stop=%0d", ret_stop);

    // Get max should return non-zero
    ret_max = $coverage_get_max(1, 0, top);
    // CHECK-NOT: coverage_max=0
    $display("coverage_max=%0d", ret_max);

    // Start and stop should return the SAME value (both succeed)
    // This prevents hardcoding one to 1 and the other to something else
    // CHECK: start_eq_stop=1
    $display("start_eq_stop=%0d", ret_start == ret_stop);

    $finish;
  end
endmodule

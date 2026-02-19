// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $coverage_control returns success indicator (1) for start/stop.
// Bug: All coverage system tasks return 0 unconditionally.
// IEEE 1800-2017 Section 20.13: $coverage_control returns 1 on success.
//
// Tests different control operations to ensure they all return success.
module top;
  integer ret_start, ret_stop, ret_reset;

  initial begin
    // Coverage start (type=1, control=1)
    ret_start = $coverage_control(1, 1, 0, top);
    // CHECK: cov_start=1
    $display("cov_start=%0d", ret_start);

    // Coverage stop (type=2, control=1)
    ret_stop = $coverage_control(2, 1, 0, top);
    // CHECK: cov_stop=1
    $display("cov_stop=%0d", ret_stop);

    // Coverage reset (type=3, control=1)
    ret_reset = $coverage_control(3, 1, 0, top);
    // CHECK: cov_reset=1
    $display("cov_reset=%0d", ret_reset);

    // All should be non-zero (1 = success)
    // CHECK: all_success=1
    $display("all_success=%0d", (ret_start != 0) && (ret_stop != 0) && (ret_reset != 0));

    $finish;
  end
endmodule

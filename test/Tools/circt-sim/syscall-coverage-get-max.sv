// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $coverage_get_max returns a non-zero value.
// Bug: All coverage system functions return 0 unconditionally.
// IEEE 1800-2017 Section 20.13: $coverage_get_max returns the maximum
// possible coverage value for the given coverage type.
module top;
  integer max_val, get_val;

  initial begin
    // $coverage_get_max should return a meaningful max value
    max_val = $coverage_get_max(1, 0, top);
    // CHECK: max_nonzero=1
    $display("max_nonzero=%0d", max_val != 0);

    // $coverage_get should return 0 or some value (≤ max)
    get_val = $coverage_get(1, 0, top);
    // At minimum, get should be <= max
    // CHECK: get_le_max=1
    $display("get_le_max=%0d", get_val <= max_val);

    $finish;
  end
endmodule

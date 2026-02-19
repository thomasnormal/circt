// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $increment for both static and dynamic arrays.
// Bug: $increment for dynamic types was stubbed to return +1 instead of -1.
// IEEE 1800-2017 Section 7.11.1.
module top;
  logic [7:0] ascending [0:3];     // ascending range: left <= right
  logic [7:0] descending [3:0];    // descending range: left >= right
  int dyn_arr[];                    // dynamic array — exercises our code path

  initial begin
    dyn_arr = new[4];

    // Static arrays: slang constant-folds these
    // For [0:3]: ascending → returns -1
    // CHECK: inc_ascending=-1
    $display("inc_ascending=%0d", $increment(ascending));

    // For [3:0]: descending → returns 1
    // CHECK: inc_descending=1
    $display("inc_descending=%0d", $increment(descending));

    // Dynamic array: this exercises our ImportVerilog code path
    // Dynamic arrays have ascending indices [0:N-1], so $increment = -1
    // Old broken code returned +1 here.
    // CHECK: inc_dynamic=-1
    $display("inc_dynamic=%0d", $increment(dyn_arr));

    $finish;
  end
endmodule

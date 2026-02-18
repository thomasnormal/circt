// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $get_coverage returns real coverage percentage from runtime
module top;
  real cov;
  initial begin
    cov = $get_coverage();
    // With no covergroups registered, coverage should be 0.0
    // CHECK: coverage=0
    $display("coverage=%0g", cov);
    $finish;
  end
endmodule

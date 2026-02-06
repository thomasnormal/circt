// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top string_methods_tb 2>&1 | FileCheck %s

// Test: basic string method operations.
// Regression test for string method interceptors.

// CHECK: len=5
// CHECK: upper=HELLO
// CHECK: lower=hello
// CHECK: sub=ell
// CHECK: [circt-sim] Simulation completed
module string_methods_tb();
  string s;

  initial begin
    s = "Hello";
    $display("len=%0d", s.len());
    $display("upper=%s", s.toupper());
    $display("lower=%s", s.tolower());
    $display("sub=%s", s.substr(1, 3));
  end
endmodule

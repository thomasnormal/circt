// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top format_char_tb 2>&1 | FileCheck %s

// Test: %c format specifier for character output.
// Regression test for moore.fmt.char support.

// CHECK: char A=A
// CHECK: char newline ok
// CHECK: [circt-sim] Simulation completed
module format_char_tb();
  initial begin
    $display("char A=%c", 8'd65);
    $write("char newline ok\n");
  end
endmodule

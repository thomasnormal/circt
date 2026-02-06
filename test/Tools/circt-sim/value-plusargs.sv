// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: env CIRCT_UVM_ARGS="+MY_INT=42 +MY_HEX=1F +MY_STR=hello" circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test $value$plusargs for integer and string extraction.
// This exercises the __moore_value_plusargs interceptor.

module top;
  int my_int;
  int my_hex;
  int found;

  initial begin
    // Test integer plusarg with %d format
    found = $value$plusargs("MY_INT=%d", my_int);
    // CHECK: found_int = 1
    $display("found_int = %0d", found);
    // CHECK: my_int = 42
    $display("my_int = %0d", my_int);

    // Test hex plusarg with %h format
    found = $value$plusargs("MY_HEX=%h", my_hex);
    // CHECK: found_hex = 1
    $display("found_hex = %0d", found);
    // CHECK: my_hex = 31
    $display("my_hex = %0d", my_hex);

    // Test missing plusarg
    found = $value$plusargs("MISSING=%d", my_int);
    // CHECK: found_missing = 0
    $display("found_missing = %0d", found);

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule

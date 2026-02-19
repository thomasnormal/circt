// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && CIRCT_UVM_ARGS="+FOO_BAR" circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $test$plusargs with a dynamically constructed string.
// Bug: $test$plusargs only works with compile-time constant strings.
// When the argument is built at runtime (e.g., via string concatenation),
// the lookup always returns 0 even though the plusarg exists.
module top;
  initial begin
    string prefix;
    string full;

    // Static string works
    // CHECK: static_found=1
    if ($test$plusargs("FOO_BAR"))
      $display("static_found=1");
    else
      $display("static_found=0");

    // Dynamic string should also work but doesn't
    prefix = "FOO";
    full = {prefix, "_BAR"};
    // CHECK: dynamic_found=1
    if ($test$plusargs(full))
      $display("dynamic_found=1");
    else
      $display("dynamic_found=0");

    $finish;
  end
endmodule

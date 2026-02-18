// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top +VERBOSE +DEBUG=7 2>&1 | FileCheck %s

module top;
  int value;
  int found;

  initial begin
    if ($test$plusargs("VERBOSE"))
      $display("VERBOSE_OK");
    else
      $display("VERBOSE_MISS");

    found = $value$plusargs("DEBUG=%d", value);
    $display("DEBUG_FOUND=%0d VAL=%0d", found, value);
    $finish;
  end

  // CHECK: VERBOSE_OK
  // CHECK: DEBUG_FOUND=1 VAL=7
  // CHECK-NOT: VERBOSE_MISS
endmodule

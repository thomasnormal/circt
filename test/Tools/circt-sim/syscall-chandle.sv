// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test chandle type (opaque pointer for DPI)
module top;
  chandle h1, h2;

  initial begin
    h1 = null;
    // CHECK: null_check=1
    $display("null_check=%0d", h1 == null);

    // Verify chandle comparison works
    h2 = null;
    // CHECK: both_null=1
    $display("both_null=%0d", h1 == h2);

    // Verify $typename shows chandle type
    // CHECK: type=chandle
    $display("type=%s", $typename(h1));

    $finish;
  end
endmodule

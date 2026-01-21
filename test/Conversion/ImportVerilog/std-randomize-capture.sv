// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test std::randomize on module-level variables
module std_randomize_capture;
  logic [7:0] val;

  // CHECK-LABEL: moore.module @std_randomize_capture
  // CHECK: %val = moore.variable : <l8>
  initial begin
    int success;
    // CHECK: moore.std_randomize %val : !moore.ref<l8>
    success = std::randomize(val);
  end
endmodule

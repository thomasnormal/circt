// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test std::randomize() scope randomization function
// IEEE 1800-2017 Section 18.12

module test;
  int x, y;
  logic [7:0] data;

  initial begin
    // CHECK: moore.std_randomize %x : !moore.ref<i32>
    if (std::randomize(x)) $display("Single var randomized");

    // CHECK: moore.std_randomize %x, %y : !moore.ref<i32>, !moore.ref<i32>
    if (std::randomize(x, y)) $display("Two vars randomized");

    // CHECK: moore.std_randomize %x, %y, %data : !moore.ref<i32>, !moore.ref<i32>, !moore.ref<l8>
    if (std::randomize(x, y, data)) $display("Three vars randomized");
  end
endmodule

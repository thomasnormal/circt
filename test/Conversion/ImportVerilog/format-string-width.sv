// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test string format specifier with width
// IEEE 1800-2017 Section 21.2.1.2 "Format specifications"

// CHECK-LABEL: moore.module @test_string_format_width
module test_string_format_width;
  string name;
  string result;

  initial begin
    name = "test";

    // Test %20s - right-justified string with width 20
    // CHECK: moore.fmt.string %{{.*}}, width 20, alignment right, padding space
    result = $sformatf("%20s", name);

    // Test %-20s - left-justified string with width 20
    // CHECK: moore.fmt.string %{{.*}}, width 20, alignment left, padding space
    result = $sformatf("%-20s", name);

    // Test %s - no width
    // CHECK: moore.fmt.string %{{.*$}}
    // CHECK-NOT: width
    result = $sformatf("%s", name);
  end
endmodule

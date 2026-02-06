// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top string_array_init_tb 2>&1 | FileCheck %s

// Test: string array initialization and comparison.
// Regression test for MooreToCore string array initializer bug where
// array elements were zero-initialized instead of using the provided values.

// CHECK: elem0=111
// CHECK: elem1=222
// CHECK: found at 1
// CHECK: [circt-sim] Simulation completed
module string_array_init_tb();
  string test [4] = '{"111", "222", "333", "444"};

  initial begin
    $display("elem0=%s", test[0]);
    $display("elem1=%s", test[1]);

    // While loop with string comparison
    begin
      int i = 0;
      while (test[i] != "222") begin
        i++;
      end
      $display("found at %0d", i);
    end
  end
endmodule

// RUN: circt-verilog %s -o %t.mlir
// RUN: circt-sim %t.mlir | FileCheck %s
// REQUIRES: slang

// Test static associative arrays persist values across function calls.
// Static associative arrays (class static variables) need runtime allocation
// via a global constructor that calls __moore_assoc_create, similar to how
// local variables are handled in VariableOpConversion.

class Container;
  static int arr[string];

  static function void add(string key, int value);
    arr[key] = value;
  endfunction

  static function int get(string key);
    if (arr.exists(key)) begin
      return arr[key];
    end else begin
      return -1;
    end
  endfunction
endclass

module test;
  initial begin
    int val;

    // Add some entries
    Container::add("first", 111);
    Container::add("second", 222);

    // Get entries - these should persist across function calls
    val = Container::get("first");
    // CHECK: Got first: 111
    $display("Got first: %0d", val);

    val = Container::get("second");
    // CHECK: Got second: 222
    $display("Got second: %0d", val);

    // Non-existent key should return -1
    val = Container::get("third");
    // CHECK: Got third: -1
    $display("Got third: %0d", val);

    // CHECK: Test passed
    $display("Test passed");
    $finish;
  end
endmodule

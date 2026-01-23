// RUN: circt-verilog %s -o - 2>&1 | FileCheck %s

// Test wildcard associative array [*] lowering to LLVM IR
// This tests the MooreToCore conversion for WildcardAssocArrayType

module top;
  // Wildcard associative array declaration and allocation
  // CHECK: llvm.call @__moore_assoc_create
  int array [*] = '{default:255};
  int val;

  initial begin
    // Write to wildcard associative array
    // CHECK: llvm.call @__moore_assoc_get_ref
    array[1] = 0;

    // Read from wildcard associative array
    // CHECK: llvm.call @__moore_assoc_get
    val = array[1];

    // Increment value
    // CHECK: llvm.call @__moore_assoc_get_ref
    array[1] = array[1] + 1;

    $display("array[1] = %0d", array[1]);
    $finish;
  end
endmodule

// CHECK-NOT: error:
// CHECK-NOT: failed to legalize

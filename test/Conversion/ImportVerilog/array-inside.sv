// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test: Unpacked arrays in 'inside' expressions
// This tests that dynamic arrays can be used as the right-hand side of 'inside'
// expressions, which generates moore.array.contains operations.

// CHECK-LABEL: moore.module @ArrayInsideTest
module ArrayInsideTest;
  int values[5];
  int target;
  int result;

  // CHECK: moore.array.contains
  initial begin
    result = target inside { values };
  end
endmodule

// Test with constraint block
// CHECK-LABEL: moore.class.classdecl @ArrayInsideConstraint
class ArrayInsideConstraint;
  rand int addr;
  int valid_addrs[];

  // CHECK: moore.constraint.block @c_valid
  // CHECK:   moore.array.contains
  constraint c_valid {
    addr inside {valid_addrs};
  }

  function new();
  endfunction
endclass

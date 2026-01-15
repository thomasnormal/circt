// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test enum iteration methods: first(), next(), last(), prev()
// IEEE 1800-2017 Section 6.19.5

typedef enum {A, B, C, D} my_enum;

// CHECK-LABEL: moore.module @test_enum_first
module test_enum_first;
  my_enum e;
  initial begin
    // first() returns the first declared enum value
    // CHECK: moore.constant 0 : i32
    e = e.first();
  end
endmodule

// CHECK-LABEL: moore.module @test_enum_last
module test_enum_last;
  my_enum e;
  initial begin
    // last() returns the last declared enum value
    // CHECK: moore.constant 3 : i32
    e = e.last();
  end
endmodule

// CHECK-LABEL: moore.module @test_enum_next
module test_enum_next;
  my_enum e;
  initial begin
    // next() returns the next enum value (with wraparound)
    // This generates a conditional chain
    // CHECK: moore.conditional
    e = e.next();
  end
endmodule

// CHECK-LABEL: moore.module @test_enum_prev
module test_enum_prev;
  my_enum e;
  initial begin
    // prev() returns the previous enum value (with wraparound)
    // CHECK: moore.conditional
    e = e.prev();
  end
endmodule

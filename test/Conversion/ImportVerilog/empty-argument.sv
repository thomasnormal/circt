// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test for EmptyArgument handling in system calls with optional arguments.
// This tests that optional/missing arguments in system calls like $random()
// (no seed) and $urandom_range(max) (no min) are properly handled.

// CHECK-LABEL: moore.module @TestEmptyArguments
module TestEmptyArguments;
  int a, b, c;
  logic [31:0] x, y;

  initial begin
    // $random() with no seed - the optional seed argument is an EmptyArgument
    // CHECK: [[V1:%[0-9]+]] = moore.builtin.random{{$}}
    // CHECK-NEXT: moore.blocking_assign %a, [[V1]]
    a = $random();

    // $urandom() with no seed - the optional seed argument is an EmptyArgument
    // CHECK: [[V2:%[0-9]+]] = moore.builtin.urandom{{$}}
    // CHECK-NEXT: moore.blocking_assign %b, [[V2]]
    b = $urandom();

    // $urandom_range(max) with no min - the min argument is an EmptyArgument
    // CHECK: [[V3:%[0-9]+]] = moore.builtin.urandom_range %{{[0-9]+}}{{$}}
    // CHECK-NEXT: moore.blocking_assign %c, [[V3]]
    c = $urandom_range(100);

    // $urandom_range with both arguments (for comparison)
    // CHECK: [[V4:%[0-9]+]] = moore.builtin.urandom_range %{{[0-9]+}}, %{{[0-9]+}}
    // CHECK-NEXT: moore.blocking_assign %c, [[V4]]
    c = $urandom_range(100, 50);

    // $random with explicit seed (for comparison)
    // CHECK: [[V5:%[0-9]+]] = moore.builtin.random seed %{{[0-9]+}}
    // CHECK-NEXT: moore.blocking_assign %a, [[V5]]
    a = $random(x);

    // $urandom with explicit seed (for comparison)
    // CHECK: [[V6:%[0-9]+]] = moore.builtin.urandom seed %{{[0-9]+}}
    // CHECK-NEXT: moore.blocking_assign %b, [[V6]]
    b = $urandom(y);
  end
endmodule

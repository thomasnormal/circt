// RUN: circt-verilog --language-version 1800-2023 --ir-moore %s | FileCheck %s

class DistDefault2023;
  rand bit [3:0] x;
  rand bit signed [3:0] y;

  // CHECK-LABEL: moore.constraint.block @c_unsigned
  constraint c_unsigned {
    // CHECK: moore.constraint.dist %{{.*}}, [1, 3], [5], [0] : !moore.i32 {default_per_range = 1 : i64, default_weight = 2 : i64}
    x dist { [1:3] := 5, default :/ 2 };
  }

  // CHECK-LABEL: moore.constraint.block @c_signed
  constraint c_signed {
    // CHECK: moore.constraint.dist %{{.*}}, [-2, 1], [7], [1] : !moore.i32 {default_per_range = 1 : i64, default_weight = 3 : i64, isSigned}
    y dist { [-2:1] :/ 7, default :/ 3 };
  }
endclass

module top;
  DistDefault2023 d;
  initial d = new;
endmodule

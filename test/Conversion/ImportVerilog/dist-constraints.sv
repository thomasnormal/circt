// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test distribution constraint parsing and IR generation

class DistTest;
  rand bit [7:0] value;
  rand bit [3:0] mode;

  // Simple distribution with single values and := weights
  // CHECK: moore.constraint.block @c_simple
  constraint c_simple {
    // CHECK: moore.constraint.dist %{{.*}}, [0, 0, 1, 1, 2, 2], [10, 20, 30], [0, 0, 0]
    value dist { 0 := 10, 1 := 20, 2 := 30 };
  }

  // Distribution with ranges
  // CHECK: moore.constraint.block @c_range
  constraint c_range {
    // CHECK: moore.constraint.dist %{{.*}}, [0, 10, 11, 20], [50, 50], [0, 0]
    value dist { [0:10] := 50, [11:20] := 50 };
  }

  // Distribution with :/ (per-range weight)
  // CHECK: moore.constraint.block @c_per_range
  constraint c_per_range {
    // CHECK: moore.constraint.dist %{{.*}}, [1, 5], [100], [1]
    mode dist { [1:5] :/ 100 };
  }

  // Distribution with unbounded range
  // CHECK: moore.constraint.block @c_unbounded
  // CHECK: moore.constraint.dist %{{.*}}, [11, 4294967295], [1], [0]
  constraint c_unbounded {
    value dist { [11:$] := 1 };
  }

  // Mixed distribution
  // CHECK: moore.constraint.block @c_mixed
  constraint c_mixed {
    // Single value with :=, range with :/
    // CHECK: moore.constraint.dist %{{.*}}, [0, 0, 1, 10], [1, 4], [0, 1]
    mode dist { 0 := 1, [1:10] :/ 4 };
  }
endclass

module top;
  DistTest dt;
  initial begin
    dt = new;
  end
endmodule

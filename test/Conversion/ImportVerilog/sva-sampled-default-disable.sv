// RUN: circt-verilog --no-uvm-auto-include %s --parse-only | FileCheck %s

module sampled_default_disable(input logic clk, reset, a);
  default clocking @(posedge clk); endclocking
  default disable iff (reset);

  property p;
    $rose(a);
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @sampled_default_disable
// CHECK: moore.procedure always
// CHECK: moore.wait_event
// CHECK: moore.blocking_assign

module sampled_default_clocking_only(input logic clk, a);
  default clocking @(posedge clk); endclocking

  property p;
    $rose(a);
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @sampled_default_clocking_only
// CHECK: moore.past
// CHECK-NOT: moore.procedure always

module sampled_explicit_clock_in_assert(input logic clk, fast, a);
  property p;
    @(posedge clk) $rose(a, @(posedge fast));
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @sampled_explicit_clock_in_assert
// CHECK: moore.procedure always
// CHECK: moore.wait_event

module sampled_explicit_same_clock_in_assert(input logic clk, a);
  property p;
    @(posedge clk) $rose(a, @(posedge clk));
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @sampled_explicit_same_clock_in_assert
// CHECK: moore.past
// CHECK-NOT: moore.procedure always

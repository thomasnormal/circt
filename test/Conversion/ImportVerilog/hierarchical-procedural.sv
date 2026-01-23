// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test hierarchical references in procedural blocks (always, initial, etc.)
// These require the hierarchical name collector to traverse procedural statements.

// CHECK-LABEL: moore.module private @Child(out q : !moore.ref<l1>)
module Child;
  // CHECK: %q = moore.variable : <l1>
  logic q;
  // CHECK: moore.procedure always_comb
  always_comb q = 1;
  // CHECK: moore.output %q : !moore.ref<l1>
endmodule

// CHECK-LABEL: moore.module @Parent()
module Parent;
  // CHECK: %u_child.q = moore.instance "u_child" @Child() -> (q: !moore.ref<l1>)
  Child u_child();

  // Hierarchical reference in initial block (common in testbenches)
  // CHECK: moore.procedure initial
  initial begin
    // CHECK: moore.read %u_child.q
    $display("q = %b", u_child.q);
  end
endmodule

// -----

// Test hierarchical references from parent to child's internal variable
// This is the force/release pattern.

// CHECK-LABEL: moore.module private @Flop(in %clk : !moore.l1, in %d : !moore.l1, out q : !moore.ref<l1>)
module Flop(input clk, d);
  // CHECK: %q = moore.variable : <l1>
  logic q;
  always @(posedge clk)
    q <= d;
  // CHECK: moore.output %q : !moore.ref<l1>
endmodule

// CHECK-LABEL: moore.module @Top(in %clk : !moore.l1, in %d : !moore.l1, in %force_val : !moore.l1, in %do_force : !moore.l1)
module Top(input clk, d, force_val, do_force);
  // CHECK: %u_flop.q = moore.instance "u_flop" @Flop(clk: {{.*}}, d: {{.*}}) -> (q: !moore.ref<l1>)
  Flop u_flop(clk, d);

  // Hierarchical reference to child's internal variable in procedural block
  // This tests the fix for hierarchical references in always blocks
  // CHECK: moore.procedure always
  always @(posedge do_force) begin
    // CHECK: moore.blocking_assign %u_flop.q
    // The force statement is simplified to a blocking assignment
    force u_flop.q = force_val;
  end
endmodule

// RUN: circt-verilog --no-uvm-auto-include --verify-diagnostics %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Keep compatibility with simulators that accept these randc-constraint forms.
// This test checks warning-level diagnostics and functional IR lowering.

class randc_solve_before_compat;
  rand bit prior;
  randc int seq;

  constraint c_impl { prior -> seq == 0; }
  // expected-warning @below {{'randc' variables cannot be used in 'solve before' constraints}}
  constraint c_order { solve prior before seq; }
endclass

class randc_soft_compat;
  randc int value;

  // expected-warning @below {{'randc' variables cannot be used in 'soft' constraints}}
  constraint c_soft_lo { soft value > 4; }
  // expected-warning @below {{'randc' variables cannot be used in 'soft' constraints}}
  constraint c_soft_hi { soft value < 12; }
endclass

// CHECK-LABEL: moore.class.classdecl @randc_solve_before_compat
// CHECK: moore.class.propertydecl @prior : !moore.i1 rand_mode rand
// CHECK: moore.class.propertydecl @seq : !moore.i32 rand_mode randc
// CHECK: moore.constraint.block @c_impl {
// CHECK: moore.constraint.implication
// CHECK: moore.constraint.expr
// CHECK: moore.constraint.block @c_order {
// CHECK: moore.constraint.solve_before [@prior], [@seq]

// CHECK-LABEL: moore.class.classdecl @randc_soft_compat
// CHECK: moore.class.propertydecl @value : !moore.i32 rand_mode randc
// CHECK: moore.constraint.block @c_soft_lo {
// CHECK: moore.constraint.expr {{.*}} : i1 soft
// CHECK: moore.constraint.block @c_soft_hi {
// CHECK: moore.constraint.expr {{.*}} : i1 soft

// CHECK-LABEL: moore.module @randc_constraint_compat_top
// CHECK: moore.randomize
// CHECK: moore.randomize
module randc_constraint_compat_top;
  initial begin
    automatic randc_solve_before_compat a = new();
    automatic randc_soft_compat b = new();
    void'(a.randomize());
    void'(b.randomize());
  end
endmodule

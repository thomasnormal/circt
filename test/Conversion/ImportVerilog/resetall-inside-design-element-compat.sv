// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s >/dev/null
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Keep compatibility with simulators that permit in-design `resetall usage.
// This checks that lowering continues across the directive with warning-level
// diagnostics and keeps subsequent behavior intact.

// CHECK-LABEL: moore.module @resetall_inside_design_element_compat
// CHECK: moore.procedure initial {
// CHECK: moore.blocking_assign %q, %{{.*}} : l1
// CHECK: moore.not
// CHECK: moore.blocking_assign %q, %{{.*}} : l1
// CHECK: moore.case_ne
// CHECK: moore.builtin.severity fatal
module resetall_inside_design_element_compat;
  logic q;

  initial begin
    q = 1'b0;
`resetall
    q = ~q;
    if (q !== 1'b1)
      $fatal(1, "q mismatch after resetall");
  end
endmodule

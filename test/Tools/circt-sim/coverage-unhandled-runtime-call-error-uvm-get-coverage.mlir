// RUN: circt-sim %s --top top 2>&1 | FileCheck %s
//
// Regression: unhandled UVM coverage runtime symbols must fail loudly instead
// of silently returning default/X values.
//
// CHECK: error: unhandled coverage runtime call in interpreter: __moore_uvm_get_coverage
// CHECK-NOT: AFTER

module {
  llvm.func @__moore_uvm_get_coverage() -> f64

  hw.module @top() {
    %fmt_after = sim.fmt.literal "AFTER"

    llhd.process {
      %cov = llvm.call @__moore_uvm_get_coverage() : () -> f64
      sim.proc.print %fmt_after
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}

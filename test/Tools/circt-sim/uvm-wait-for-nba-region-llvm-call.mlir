// RUN: circt-sim %s 2>&1 | FileCheck %s
//
// Regression: ensure uvm_wait_for_nba_region interception works for direct
// llvm.call sites in UVM phasing code.
//
// CHECK: after_llvm_wait
// CHECK-NOT: failed

llvm.func @"uvm_pkg::uvm_wait_for_nba_region"()

hw.module @top() {
  %after = sim.fmt.literal "after_llvm_wait"
  %nl = sim.fmt.literal "\0A"

  llhd.process {
    llvm.call @"uvm_pkg::uvm_wait_for_nba_region"() : () -> ()
    %line = sim.fmt.concat (%after, %nl)
    sim.proc.print %line
    llhd.halt
  }

  hw.output
}

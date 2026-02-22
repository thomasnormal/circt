// RUN: env CIRCT_SIM_PROFILE_FUNCS=1 CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 circt-sim %s 2>&1 | FileCheck %s
//
// Verify optional process/profile summary emission on normal simulation exit.
//
// CHECK: [circt-sim] Process states:
// CHECK: proc 1

module {
  hw.module @top() {
    %fmt = sim.fmt.literal "done\0A"
    llhd.process {
      sim.proc.print %fmt
      llhd.halt
    }
    hw.output
  }
}

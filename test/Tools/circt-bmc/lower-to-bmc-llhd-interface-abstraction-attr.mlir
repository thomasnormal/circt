// RUN: circt-opt --lower-to-bmc="bound=4 top-module=top" %s | FileCheck %s

module {
  hw.module @top(in %in : i1) attributes {num_regs = 0 : i32, initial_values = [], circt.bmc_abstracted_llhd_interface_inputs = 5 : i32} {
    verif.assert %in : i1
    hw.output
  }
}

// CHECK: verif.bmc bound 8 num_regs 0 initial_values [] attributes {{{.*}}bmc_abstracted_llhd_interface_inputs = 5 : i32{{.*}}} init {

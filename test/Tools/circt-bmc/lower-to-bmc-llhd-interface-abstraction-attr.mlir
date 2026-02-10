// RUN: circt-opt --lower-to-bmc="bound=4 top-module=top" %s | FileCheck %s

module {
  hw.module @top(in %in : i1) attributes {num_regs = 0 : i32, initial_values = [], circt.bmc_abstracted_llhd_interface_inputs = 5 : i32, circt.bmc_abstracted_llhd_interface_input_details = [{name = "sig_field0_unknown", base = "sig_field0_unknown", type = i1}]} {
    verif.assert %in : i1
    hw.output
  }
}

// CHECK: verif.bmc bound 8 num_regs 0 initial_values [] attributes {
// CHECK: bmc_abstracted_llhd_interface_input_details =
// CHECK: sig_field0_unknown
// CHECK: bmc_abstracted_llhd_interface_inputs = 5 : i32
// CHECK: init {

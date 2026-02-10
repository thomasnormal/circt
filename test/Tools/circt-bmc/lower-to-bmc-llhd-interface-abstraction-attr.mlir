// RUN: circt-opt --lower-to-bmc="bound=4 top-module=top" %s | FileCheck %s
// RUN: circt-opt --lower-to-bmc="bound=4 top-module=top" %s 2>&1 | FileCheck %s --check-prefix=WARN

module {
  hw.module @top(in %in : i1) attributes {num_regs = 0 : i32, initial_values = [], circt.bmc_abstracted_llhd_interface_inputs = 5 : i32, circt.bmc_abstracted_llhd_interface_input_details = [{name = "sig_field0_unknown", base = "sig_field0_unknown", type = i1, reason = "interface_enable_resolution_unknown", signal = "sig", field = 0 : i64, loc = "dummy"}]} {
    verif.assert %in : i1
    hw.output
  }
}

// CHECK: verif.bmc bound 8 num_regs 0 initial_values [] attributes {
// CHECK: bmc_abstracted_llhd_interface_input_details =
// CHECK-DAG: sig_field0_unknown
// CHECK-DAG: interface_enable_resolution_unknown
// CHECK-DAG: signal = "sig"
// CHECK-DAG: field = 0
// CHECK: bmc_abstracted_llhd_interface_inputs = 5 : i32
// CHECK: init {

// WARN: BMC_PROVENANCE_LLHD_INTERFACE reason=interface_enable_resolution_unknown signal=sig field=0 name=sig_field0_unknown

// RUN: circt-opt --lower-to-bmc="bound=4 top-module=top" %s | FileCheck %s
// RUN: circt-opt --lower-to-bmc="bound=4 top-module=top" %s 2>&1 | FileCheck %s --check-prefix=WARN

module {
  hw.module @top(in %in : i1) attributes {num_regs = 0 : i32, initial_values = [], circt.bmc_abstracted_llhd_process_results = 3 : i32, circt.bmc_abstracted_llhd_process_result_details = [{name = "llhd_process_result", base = "llhd_process_result", type = i1, reason = "observable_signal_use", signal = "sig", result = 0 : i64, loc = "dummy"}]} {
    verif.assert %in : i1
    hw.output
  }
}

// CHECK: verif.bmc bound 8 num_regs 0 initial_values [] attributes {
// CHECK: bmc_abstracted_llhd_process_result_details =
// CHECK-DAG: llhd_process_result
// CHECK-DAG: observable_signal_use
// CHECK-DAG: signal = "sig"
// CHECK-DAG: result = 0
// CHECK: bmc_abstracted_llhd_process_results = 3 : i32
// CHECK: init {

// WARN: BMC_PROVENANCE_LLHD_PROCESS reason=observable_signal_use result=0 signal=sig name=llhd_process_result

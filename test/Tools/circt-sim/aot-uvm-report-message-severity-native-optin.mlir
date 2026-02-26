// RUN: circt-sim-compile %s -o %t.default.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 circt-sim-compile %s -o %t.optin.so 2>&1 | FileCheck %s --check-prefix=OPTIN

// DEFAULT: [circt-sim-compile] Demoted 1 intercepted functions to trampolines
// DEFAULT: [circt-sim-compile] 1 functions + 0 processes ready for codegen

// OPTIN-NOT: Demoted 1 intercepted functions to trampolines
// OPTIN: [circt-sim-compile] 2 functions + 0 processes ready for codegen

func.func private @"uvm_pkg::uvm_report_message::get_severity"() -> i32 {
  %c0 = arith.constant 0 : i32
  return %c0 : i32
}

func.func private @keep_alive() -> i32 {
  %v = call @"uvm_pkg::uvm_report_message::get_severity"() : () -> i32
  return %v : i32
}

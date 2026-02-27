// RUN: circt-compile %s -o %t.default.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 circt-compile %s -o %t.optin.so 2>&1 | FileCheck %s --check-prefix=OPTIN
// RUN: circt-sim %s --top test --compiled=%t.default.so 2>&1 | FileCheck %s --check-prefix=RUN-DEFAULT
// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 circt-sim %s --top test --compiled=%t.optin.so 2>&1 | FileCheck %s --check-prefix=RUN-OPTIN

// Regression: keep report_message::get_severity native-enabled by default.
// This used to be demoted as a tactical guard before assoc pointer
// normalization landed in the Moore runtime.
//
// DEFAULT-NOT: Demoted 1 intercepted functions to trampolines
// DEFAULT: [circt-compile] 2 functions + 0 processes ready for codegen
//
// OPTIN-NOT: Demoted 1 intercepted functions to trampolines
// OPTIN: [circt-compile] 2 functions + 0 processes ready for codegen
//
// RUN-DEFAULT: sev=0
// RUN-OPTIN: sev=0

func.func private @"uvm_pkg::uvm_report_message::get_severity"() -> i32 {
  %c0 = arith.constant 0 : i32
  return %c0 : i32
}

func.func private @keep_alive() -> i32 {
  %v = call @"uvm_pkg::uvm_report_message::get_severity"() : () -> i32
  return %v : i32
}

hw.module @test() {
  %c5_i64 = hw.constant 5000000 : i64
  %c15_i64 = hw.constant 15000000 : i64
  %fmt_prefix = sim.fmt.literal "sev="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @keep_alive() : () -> i32
    %fmt_val = sim.fmt.dec %r : i32
    %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c15_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}

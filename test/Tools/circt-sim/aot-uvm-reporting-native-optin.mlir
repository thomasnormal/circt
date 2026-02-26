// RUN: circt-sim-compile %s -o %t.default.so 2>&1 | FileCheck %s --check-prefix=COMPILE-DEFAULT
// RUN: circt-sim %s --compiled=%t.default.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 circt-sim-compile %s -o %t.optin.so 2>&1 | FileCheck %s --check-prefix=COMPILE-OPTIN
// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 circt-sim %s --compiled=%t.optin.so 2>&1 | FileCheck %s --check-prefix=OPTIN

// Regression: keep UVM reporting handler/object paths interpreted by default.
//
// COMPILE-DEFAULT: [circt-sim-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE-DEFAULT: [circt-sim-compile] Demoted 1 intercepted functions to trampolines
// COMPILE-DEFAULT: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// DEFAULT: Loaded 1 compiled functions: 1 native-dispatched, 0 not-native-dispatched, 0 intercepted
// DEFAULT: out=10
//
// COMPILE-OPTIN: [circt-sim-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE-OPTIN: [circt-sim-compile] 2 functions + 0 processes ready for codegen
//
// OPTIN: Loaded 2 compiled functions: 2 native-dispatched, 0 not-native-dispatched, 0 intercepted
// OPTIN: out=10

func.func @"uvm_pkg::uvm_report_handler::process_report_message"(%a: i32, %b: i32) -> i32 {
  %r = arith.addi %a, %b : i32
  return %r : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

hw.module @test() {
  %c3_i32 = hw.constant 3 : i32
  %c7_i32 = hw.constant 7 : i32
  %c5_i64 = hw.constant 5000000 : i64
  %c15_i64 = hw.constant 15000000 : i64
  %fmt_prefix = sim.fmt.literal "out="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @"uvm_pkg::uvm_report_handler::process_report_message"(%c3_i32, %c7_i32) : (i32, i32) -> i32
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

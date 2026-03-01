// RUN: circt-compile %s -o %t.default.so 2>&1 | FileCheck %s --check-prefix=COMPILE-DEFAULT
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.default.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 circt-compile %s -o %t.optin.so 2>&1 | FileCheck %s --check-prefix=COMPILE-OPTIN
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 circt-sim %s --top top --compiled=%t.optin.so 2>&1 | FileCheck %s --check-prefix=OPTIN

// Regression: reporting opt-in should be honored consistently across compile
// and runtime policy. When a reporting helper is compiled natively in opt-in
// mode, direct func.call should not be blocked by generic unmapped uvm_pkg::*
// deny policy.
//
// COMPILE-DEFAULT: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE-DEFAULT: [circt-compile] 1 functions + 0 processes ready for codegen
//
// DEFAULT: Loaded 1 compiled functions: 1 native-dispatched, 0 not-native-dispatched, 0 intercepted
// DEFAULT: Compiled function calls:          1
// DEFAULT: Interpreted function calls:       0
// DEFAULT: direct_calls_native:              1
// DEFAULT: direct_calls_interpreted:         0
// DEFAULT: Top interpreted func.call fallback reasons (top 50):
// DEFAULT: 1x uvm_pkg::uvm_report_handler::get_verbosity_level [no-native=1]
// DEFAULT: out=47
//
// COMPILE-OPTIN: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE-OPTIN: [circt-compile] 2 functions + 0 processes ready for codegen
//
// OPTIN: Loaded 2 compiled functions: 2 native-dispatched, 0 not-native-dispatched, 0 intercepted
// OPTIN: Compiled function calls:          2
// OPTIN: Interpreted function calls:       0
// OPTIN: direct_calls_native:              2
// OPTIN: direct_calls_interpreted:         0
// OPTIN: out=47

func.func @"uvm_pkg::uvm_report_handler::get_verbosity_level"(%x: i32) -> i32 {
  %c42 = arith.constant 42 : i32
  %r = arith.addi %x, %c42 : i32
  return %r : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

hw.module @top() {
  %fmtPrefix = sim.fmt.literal "out="
  %fmtNl = sim.fmt.literal "\0A"
  %c5 = hw.constant 5 : i32
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %r0 = func.call @"uvm_pkg::uvm_report_handler::get_verbosity_level"(%c5) : (i32) -> i32
    %r1 = func.call @keep_alive(%r0) : (i32) -> i32
    %fmtV = sim.fmt.dec %r1 signed : i32
    %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtV, %fmtNl)
    sim.proc.print %fmtOut
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c10_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}

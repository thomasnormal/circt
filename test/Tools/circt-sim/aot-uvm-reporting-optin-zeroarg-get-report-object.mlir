// RUN: circt-compile %s -o %t.default.so 2>&1 | FileCheck %s --check-prefix=COMPILE-DEFAULT
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.default.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 circt-compile %s -o %t.optin.so 2>&1 | FileCheck %s --check-prefix=COMPILE-OPTIN
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 circt-sim %s --top top --compiled=%t.optin.so 2>&1 | FileCheck %s --check-prefix=OPTIN-SAFE
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING_UNMAPPED_UNSAFE=1 circt-sim %s --top top --compiled=%t.optin.so 2>&1 | FileCheck %s --check-prefix=OPTIN-UNSAFE

// Regression: reporting opt-in should not be blocked by generic zero-arg
// helper safety for `uvm_get_report_object` when explicit unmapped-unsafe
// opt-in is set.
//
// COMPILE-DEFAULT: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE-DEFAULT: [circt-compile] 2 functions + 0 processes ready for codegen
//
// DEFAULT: Loaded 2 compiled functions: 2 native-dispatched, 0 not-native-dispatched, 0 intercepted
// DEFAULT: Compiled function calls:          1
// DEFAULT: Interpreted function calls:       1
// DEFAULT: direct_calls_native:              1
// DEFAULT: direct_calls_interpreted:         1
// DEFAULT: Top interpreted func.call fallback reasons (top 50):
// DEFAULT: 1x uvm_pkg::uvm_get_report_object [unmapped-policy=1]
// DEFAULT: out=47
//
// COMPILE-OPTIN: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE-OPTIN: [circt-compile] 2 functions + 0 processes ready for codegen
//
// OPTIN-SAFE: Loaded 2 compiled functions: 2 native-dispatched, 0 not-native-dispatched, 0 intercepted
// OPTIN-SAFE: Compiled function calls:          1
// OPTIN-SAFE: Interpreted function calls:       1
// OPTIN-SAFE: direct_calls_native:              1
// OPTIN-SAFE: direct_calls_interpreted:         1
// OPTIN-SAFE: Top interpreted func.call fallback reasons (top 50):
// OPTIN-SAFE: 1x uvm_pkg::uvm_get_report_object [unmapped-policy=1]
// OPTIN-SAFE: out=47
//
// OPTIN-UNSAFE: Loaded 2 compiled functions: 2 native-dispatched, 0 not-native-dispatched, 0 intercepted
// OPTIN-UNSAFE: Compiled function calls:          2
// OPTIN-UNSAFE: Interpreted function calls:       0
// OPTIN-UNSAFE: direct_calls_native:              2
// OPTIN-UNSAFE: direct_calls_interpreted:         0
// OPTIN-UNSAFE: out=47

func.func @"uvm_pkg::uvm_get_report_object"() -> i32 {
  %c5 = arith.constant 5 : i32
  return %c5 : i32
}

func.func @plus_42(%x: i32) -> i32 {
  %c42 = arith.constant 42 : i32
  %r = arith.addi %x, %c42 : i32
  return %r : i32
}

hw.module @top() {
  %fmtPrefix = sim.fmt.literal "out="
  %fmtNl = sim.fmt.literal "\0A"
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %r0 = func.call @"uvm_pkg::uvm_get_report_object"() : () -> i32
    %r1 = func.call @plus_42(%r0) : (i32) -> i32
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

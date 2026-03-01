// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 circt-compile %s -o %t.so
// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s

// Repro: when interpreted execution is entered from a trampoline (aotDepth=1),
// direct native dispatch historically demoted all nested func.call sites.
// With reporting opt-in enabled, report-handler setters are safe and should
// stay native instead of being attributed as depth fallbacks.

module {
  func.func private @driver(%x: i32) -> i32 {
    %r = func.call @"uvm_pkg::uvm_object::new"(%x) : (i32) -> i32
    return %r : i32
  }

  // Constructors are still policy-intercepted by default and therefore execute
  // through trampoline->interpreter with aotDepth=1.
  func.func private @"uvm_pkg::uvm_object::new"(%x: i32) -> i32 {
    %r = func.call @"uvm_pkg::uvm_report_handler::set_severity_action"(%x)
      : (i32) -> i32
    return %r : i32
  }

  // Under reporting opt-in this callee is native-eligible.
  func.func private @"uvm_pkg::uvm_report_handler::set_severity_action"(
      %x: i32) -> i32 {
    %c7 = arith.constant 7 : i32
    %r = arith.addi %x, %c7 : i32
    return %r : i32
  }

  hw.module @top() {
    %fmtPrefix = sim.fmt.literal "out="
    %fmtNl = sim.fmt.literal "\0A"
    %c5 = hw.constant 5 : i32
    llhd.process {
      %r = func.call @driver(%c5) : (i32) -> i32
      %fmtV = sim.fmt.dec %r signed : i32
      %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtV, %fmtNl)
      sim.proc.print %fmtOut
      llhd.halt
    }
    hw.output
  }
}

// CHECK: Loaded 2 compiled functions: 2 native-dispatched, 0 not-native-dispatched, 0 intercepted
// CHECK: Compiled function calls:          2
// CHECK: Interpreted function calls:       0
// CHECK-NOT: uvm_pkg::uvm_report_handler::set_severity_action [depth=
// CHECK: out=12

// RUN: circt-compile %s -o %t.default.so 2>&1 | FileCheck %s --check-prefix=COMPILE-DEFAULT
// RUN: circt-sim %s --compiled=%t.default.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_AGGRESSIVE_UVM=1 circt-compile %s -o %t.fast.so 2>&1 | FileCheck %s --check-prefix=COMPILE-FAST
// RUN: env CIRCT_AOT_AGGRESSIVE_UVM=1 circt-sim %s --compiled=%t.fast.so 2>&1 | FileCheck %s --check-prefix=FAST

// Regression: umbrella env enables all UVM native opt-ins.
//
// COMPILE-DEFAULT: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE-DEFAULT: [circt-compile] Demoted 1 intercepted functions to trampolines
// COMPILE-DEFAULT: [circt-compile] 1 functions + 0 processes ready for codegen
//
// DEFAULT: Loaded 1 compiled functions: 1 native-dispatched, 0 not-native-dispatched, 0 intercepted
// DEFAULT: out=47
//
// COMPILE-FAST: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE-FAST: [circt-compile] 2 functions + 0 processes ready for codegen
//
// FAST: Loaded 2 compiled functions: 2 native-dispatched, 0 not-native-dispatched, 0 intercepted
// FAST: out=47

func.func @"uvm_pkg::toy_type::get_type_name"(%a: i32) -> i32 {
  %c42 = arith.constant 42 : i32
  %r = arith.addi %a, %c42 : i32
  return %r : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

hw.module @test() {
  %c5_i32 = hw.constant 5 : i32
  %c5_i64 = hw.constant 5000000 : i64
  %c15_i64 = hw.constant 15000000 : i64
  %fmt_prefix = sim.fmt.literal "out="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @"uvm_pkg::toy_type::get_type_name"(%c5_i32) : (i32) -> i32
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

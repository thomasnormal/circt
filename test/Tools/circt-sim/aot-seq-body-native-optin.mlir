// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_ALLOW_NATIVE_SEQ_BODY=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=OPTIN

// Opt-in regression: allow native dispatch for ::body methods.
//
// DEFAULT: Loaded 1 compiled functions: 0 native-dispatched, 0 not-native-dispatched, 1 intercepted
// DEFAULT: out=200
//
// OPTIN: Loaded 1 compiled functions: 1 native-dispatched, 0 not-native-dispatched, 0 intercepted
// OPTIN: out=200

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable

func.func @"uvm_pkg::toy_sequence::body"(%a: i32, %b: i32) -> i32 {
  %c = arith.muli %a, %b : i32
  return %c : i32
}

hw.module @test() {
  %c10_i32 = hw.constant 10 : i32
  %c20_i32 = hw.constant 20 : i32
  %c5_i64 = hw.constant 5000000 : i64
  %c15_i64 = hw.constant 15000000 : i64
  %fmt_prefix = sim.fmt.literal "out="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @"uvm_pkg::toy_sequence::body"(%c10_i32, %c20_i32) : (i32, i32) -> i32
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

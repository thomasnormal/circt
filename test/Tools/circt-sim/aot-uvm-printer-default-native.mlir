// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME

// Regression: UVM printer/topology helper methods should stay native by
// default to avoid pathological startup slowdown in APB/UVM benches.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] 2 functions + 0 processes ready for codegen
//
// RUNTIME: Loaded 2 compiled functions: 2 native-dispatched, 0 not-native-dispatched, 0 intercepted
// RUNTIME: out=42

func.func @"uvm_pkg::uvm_tree_printer::emit"(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %r = arith.addi %x, %c2 : i32
  return %r : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

hw.module @top() {
  %c40 = hw.constant 40 : i32
  %c10_i64 = hw.constant 10000000 : i64
  %fmtPrefix = sim.fmt.literal "out="
  %fmtNl = sim.fmt.literal "\0A"

  llhd.process {
    %v = func.call @"uvm_pkg::uvm_tree_printer::emit"(%c40) : (i32) -> i32
    %r = func.call @keep_alive(%v) : (i32) -> i32
    %fmtV = sim.fmt.dec %r signed : i32
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

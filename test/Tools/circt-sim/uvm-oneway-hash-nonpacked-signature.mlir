// RUN: circt-sim %s | FileCheck %s
//
// Regression: the uvm_oneway_hash interceptor must only apply to canonical
// packed-string calls. Non-packed helper functions with the same symbol name
// must execute their interpreted function body.
//
// CHECK: out=42

func.func @"uvm_pkg::uvm_oneway_hash"(%a: i32, %b: i32) -> i32 {
  %r = arith.addi %a, %b : i32
  return %r : i32
}

hw.module @top() {
  %c40 = hw.constant 40 : i32
  %c2 = hw.constant 2 : i32
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"

  llhd.process {
    %r = func.call @"uvm_pkg::uvm_oneway_hash"(%c40, %c2) : (i32, i32) -> i32
    %rfmt = sim.fmt.dec %r : i32
    %msg = sim.fmt.concat (%prefix, %rfmt, %nl)
    sim.proc.print %msg
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}

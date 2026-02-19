// RUN: circt-sim %s | FileCheck %s

// Regression for native UVM size interception:
// If a port is not tracked in analysisPortConnections, circt-sim must fall
// back to the real function body instead of forcing size=0.

// CHECK: size_result=7

func.func private @"uvm_pkg::uvm_port_base::size"(%self: i64) -> i32 {
  %c7 = hw.constant 7 : i32
  return %c7 : i32
}

hw.module @top() {
  llhd.process {
    %self = hw.constant 4660 : i64
    %sz = func.call @"uvm_pkg::uvm_port_base::size"(%self) : (i64) -> i32

    %lit = sim.fmt.literal "size_result="
    %val = sim.fmt.dec %sz signed : i32
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.concat (%lit, %val, %nl)
    sim.proc.print %fmt
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}

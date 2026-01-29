// RUN: circt-sim %s --max-time=1000000 | FileCheck %s

// Test arith.select on format strings
// This ensures conditional printing (if-then-else with $display) works correctly

module {
  hw.module @test() {
    %true_val = hw.constant 1 : i1
    %false_val = hw.constant 0 : i1
    %0 = sim.fmt.literal "TRUE\0A"
    %1 = sim.fmt.literal "FALSE\0A"

    llhd.process {
      // Select true branch
      %result1 = arith.select %true_val, %0, %1 : !sim.fstring
      sim.proc.print %result1

      // Select false branch
      %result2 = arith.select %false_val, %0, %1 : !sim.fstring
      sim.proc.print %result2

      llhd.halt
    }
    hw.output
  }
}

// CHECK: TRUE
// CHECK: FALSE

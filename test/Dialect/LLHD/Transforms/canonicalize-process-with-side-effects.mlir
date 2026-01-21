// RUN: circt-opt %s --canonicalize | FileCheck %s

// Test that llhd.process ops with sim.proc.print are NOT removed by canonicalization
// (even though they have no results and no DriveOp)

hw.module @test_display_preserved() {
  %fmt = sim.fmt.literal "Hello World!\0A"
  %delay = llhd.constant_time <10ns, 0d, 0e>
  // CHECK: llhd.process
  // CHECK-NEXT: sim.proc.print
  // CHECK-NEXT: llhd.wait
  // CHECK-NEXT: llhd.halt
  llhd.process {
    sim.proc.print %fmt
    llhd.wait delay %delay, ^exit
  ^exit:
    llhd.halt
  }
  hw.output
}

// Test that llhd.process ops with sim.terminate are NOT removed
hw.module @test_terminate_preserved() {
  // CHECK: llhd.process
  // CHECK-NEXT: sim.terminate
  // CHECK-NEXT: llhd.halt
  llhd.process {
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}

// Test that empty processes with no side effects ARE removed
hw.module @test_empty_removed() {
  %delay = llhd.constant_time <10ns, 0d, 0e>
  // CHECK-NOT: llhd.process
  llhd.process {
    llhd.wait delay %delay, ^exit
  ^exit:
    llhd.halt
  }
  hw.output
}

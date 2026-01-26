// RUN: circt-opt %s --canonicalize | FileCheck %s

// Test that llhd.process ops with sim.proc.print are NOT removed by canonicalization
// (even though they have no results and no DriveOp)

hw.module @test_display_preserved() {
  %fmt = sim.fmt.literal "Hello World!\0A"
  %delay = llhd.constant_time <10ns, 0d, 0e>
  // CHECK: llhd.process
  // CHECK:   sim.proc.print
  // CHECK:   llhd.wait
  // CHECK:   llhd.halt
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
  // CHECK:   sim.terminate
  // CHECK:   llhd.halt
  llhd.process {
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}

// Test that empty processes with no side effects ARE removed
hw.module @test_empty_removed() {
  %delay = llhd.constant_time <10ns, 0d, 0e>
  // CHECK-LABEL: @test_empty_removed
  // CHECK-NOT: llhd.process
  llhd.process {
    llhd.wait delay %delay, ^exit
  ^exit:
    llhd.halt
  }
  hw.output
}

// Test that llhd.process ops with event-based wait (observed operands) are NOT removed
// Even though they have no DriveOp or other side effects, event-based waits
// represent a reactive process that monitors signal changes.
hw.module @test_event_wait_preserved(in %sig : !llhd.ref<i8>) {
  %val = llhd.prb %sig : i8
  // CHECK-LABEL: @test_event_wait_preserved
  // CHECK: llhd.process
  // CHECK:   llhd.wait (%{{.*}} : i8)
  // CHECK:   llhd.halt
  llhd.process {
    llhd.wait (%val : i8), ^exit
  ^exit:
    llhd.halt
  }
  hw.output
}

// Test that llhd.process ops with verif.assert are NOT removed
hw.module @test_verif_assert_preserved(in %cond : i1) {
  // CHECK-LABEL: @test_verif_assert_preserved
  // CHECK: llhd.process
  // CHECK: verif.assert
  llhd.process {
    verif.assert %cond : i1
    llhd.halt
  }
  hw.output
}

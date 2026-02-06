// RUN: circt-sim %s | FileCheck %s

// Test that continuous assignments (module-level drives from probed signals)
// correctly propagate signal changes.

// This tests the fix for module instantiation signal connections where
// module-level drives were not being re-evaluated when their source signals
// changed.

// CHECK: [circt-sim] Starting simulation
// CHECK: Value updated to: 1
// CHECK: Value updated to: 2
// CHECK: Value updated to: 3
// CHECK: [circt-sim] Simulation completed

hw.module @test() {
  %true = hw.constant true
  %false = hw.constant false
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c50000000_i64 = hw.constant 50000000 : i64  // 50 time units
  %c10000000_i64 = hw.constant 10000000 : i64  // 10 time units
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %fmt_lit = sim.fmt.literal "Value updated to: "
  %fmt_nl = sim.fmt.literal "\0A"

  // Source signal - this will be incremented
  %source = llhd.sig %c0_i32 : i32

  // Target signal - should track source via continuous assignment
  %target = llhd.sig %c0_i32 : i32

  // Probe the source signal
  %source_val = llhd.prb %source : i32

  // Continuous assignment: target = source
  // This should re-execute whenever source changes
  llhd.drv %target, %source_val after %eps : i32

  // Process that increments source three times
  llhd.process {
    %delay = llhd.int_to_time %c10000000_i64

    // First increment
    cf.br ^bb1
  ^bb1:
    llhd.wait delay %delay, ^bb2
  ^bb2:
    %v1 = llhd.prb %source : i32
    %v2 = comb.add %v1, %c1_i32 : i32
    llhd.drv %source, %v2 after %delta : i32

    // Second increment
    llhd.wait delay %delay, ^bb3
  ^bb3:
    %v3 = llhd.prb %source : i32
    %v4 = comb.add %v3, %c1_i32 : i32
    llhd.drv %source, %v4 after %delta : i32

    // Third increment
    llhd.wait delay %delay, ^bb4
  ^bb4:
    %v5 = llhd.prb %source : i32
    %v6 = comb.add %v5, %c1_i32 : i32
    llhd.drv %source, %v6 after %delta : i32

    llhd.wait delay %delay, ^bb5
  ^bb5:
    llhd.halt
  }

  // Process that prints target value when it changes
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %old_val = llhd.prb %target : i32
    llhd.wait (%old_val : i32), ^bb2
  ^bb2:
    %new_val = llhd.prb %target : i32
    %changed = comb.icmp ne %old_val, %new_val : i32
    cf.cond_br %changed, ^print, ^bb1
  ^print:
    %fmt_val = sim.fmt.dec %new_val signed : i32
    %fmt_str = sim.fmt.concat (%fmt_lit, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_str
    cf.br ^bb1
  }

  // Termination process
  llhd.process {
    %timeout = llhd.int_to_time %c50000000_i64
    llhd.wait delay %timeout, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
